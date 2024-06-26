// InfoGAN.js

const InfoGAN = (function() {
    class InfoGAN {
        constructor(
            {
                latentDim = 100,
                codeDim = 3,
                genLayers = 0,
                genStartDim = 512,
                discLayers = 1,
                discStartDim = 512,
                batchSize = 64,
                qWeight = 1,
                latentNorm = 1/4,
            } = {}
        ) {
            this.generator = null;
            this.discriminator = null;
            this.qNetwork = null;
            this.combinedModel = null;
            this.latentDim = latentDim;
            this.codeDim = codeDim;
            this.genLayers = genLayers;
            this.genStartDim = genStartDim;
            this.discLayers = discLayers;
            this.discStartDim = discStartDim;
            this.batchSize = batchSize;
            this.qWeight = qWeight;
            this.latentNorm = latentNorm;
        }
    
        async init() {
            this.buildGenerator({
                latentDim: this.latentDim, 
                codeDim: this.codeDim, 
                numLayers: this.genLayers, 
                startDim: this.genStartDim
            });
            this.buildDiscriminator({
                numLayers: this.discLayers, 
                startDim: this.discStartDim
            });
            this.buildQNetwork({
                codeDim: this.codeDim, 
                numLayers: this.discLayers, 
                startDim: this.discStartDim
            });
            this.buildCombinedModel();
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.codeSamplesBuff = tf.buffer([this.batchSize]);
            this.fakeSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.isTraining = false;
        }
    
        buildGenerator({latentDim, codeDim, numLayers, startDim}={}) {
            console.log(`Building Generator with latentDim: ${latentDim}, codeDim: ${codeDim}, numLayers: ${numLayers}, startDim: ${startDim}`);

            this.gLatent = tf.input({ shape: [latentDim] });
            this.gCode = tf.input({ shape: [codeDim] });
            
            const latentEmb = tf.layers.dense({ units: startDim, activation: 'linear', useBias: false }).apply(this.gLatent);
            const codeEmb = tf.layers.dense({ units: startDim, activation: 'linear', useBias: false }).apply(this.gCode);

            // const gEmbedding = tf.layers.add().apply([latentEmb, codeEmb]);
            const normLatent = new MultiplyLayer({ constant: this.latentNorm }).apply(latentEmb);
            let gEmbedding = tf.layers.add().apply([normLatent, codeEmb]);
            gEmbedding = tf.layers.activation({ activation: 'relu' }).apply(gEmbedding);

            
            // const backbone = tf.sequential();
            // backbone.add(tf.layers.dense({ units: startDim, inputShape: [startDim], activation: 'relu' }));
            let gOutput = gEmbedding;
            for (let i = 1; i < numLayers; i++) {
                const inputShape = [startDim * Math.pow(2, i-1)];
                const units = startDim * Math.pow(2, i);
                const activation = "relu";
                const layer = tf.layers.dense({ units, inputShape, activation });
                console.log(`Layer ${i}: units: ${units}, inputShape: ${inputShape}, activation: ${activation}`)
                gOutput = layer.apply(gOutput);
            }
    
            const finalLayer = tf.layers.dense({ units: 2, activation: 'linear' });
            gOutput = finalLayer.apply(gOutput);
            // backbone.add(tf.layers.dense({ units: 2, inputShape: [startDim * Math.pow(2, numLayers-1)], activation: 'linear' }));
            // const gOutput = backbone.apply(gEmbedding);

            this.generator = tf.model({ inputs: [this.gLatent, this.gCode], outputs: gOutput });
        }
    
        buildDiscriminator({numLayers, startDim}={}) {
            const dInput = tf.input({ shape: [2] });
            const discriminator = tf.sequential();
            discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }

            discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
            const dOutput = discriminator.apply(dInput);

            this.discriminator = tf.model({ inputs: dInput, outputs: dOutput });
            this.discriminator.compile({ 
                optimizer: tf.train.adam(0.0005),
                loss: 'binaryCrossentropy' 
            });
        }
    
        buildQNetwork({codeDim, numLayers, startDim}) {
            const qInput = tf.input({ shape: [2] });
            const qNetwork = tf.sequential();
            // qNetwork.add(tf.layers.dense({ units: 128, inputShape: [2], activation: 'relu' }));
            qNetwork.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
            for (let i = 1; i < numLayers; i++) {
                qNetwork.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }

            qNetwork.add(tf.layers.dense({ units: codeDim, activation: 'softmax' }));
            const qOutput = qNetwork.apply(qInput);

            this.qNetwork = tf.model({ inputs: qInput, outputs: qOutput });
        }

        buildCombinedModel() {
            const gOutput = this.generator.outputs[0];
            const dOutput = this.discriminator.apply(gOutput);
            const qOutput = this.qNetwork.apply(gOutput);

            this.combinedModel = tf.model({ inputs: this.generator.inputs, outputs: [dOutput, qOutput] });
            // const gLossFn = (yTrue, yPred) => tf.scalar(1).mul(tf.metrics.categoricalCrossentropy(yTrue, yPred));
            const gLossFn = (yTrue, yPred) => tf.metrics.binaryCrossentropy(yTrue, yPred).mul(tf.scalar(1));
            const qLossFn = (yTrue, yPred) => tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(tf.scalar(this.qWeight));
            this.combinedModel.compile({ 
                optimizer: tf.train.adam(0.0001, 0.5, 0.5), 
                // optimizer: tf.train.momentum(0.001, 0.1),
                loss: [gLossFn, qLossFn],
            });
        }


        readTrainingBuffer() {
            const realData = this.realSamplesBuff.toTensor();
            const codeData = this.codeSamplesBuff.toTensor();
            const fakeData = this.fakeSamplesBuff.toTensor();

            const ret = {
                realData: realData.arraySync(),
                codeData: codeData.arraySync(),
                fakeData: fakeData.arraySync(),
            }
            tf.dispose([realData, codeData, fakeData]);
            return ret;
        }

        async trainToggle(data, callback = null) {
            if (this.isTraining) {
                this.isTraining = false;
                return;
            }
    
            this.isTraining = true;
            let iter = 0;
    
            const realLabels = tf.ones([this.batchSize, 1]);
            const fakeLabels = tf.zeros([this.batchSize, 1]);
            const dLabels = tf.concat([realLabels, fakeLabels]);

            while (this.isTraining) { 
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const realSamples = this.realSamplesBuff.toTensor();
                const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
                const idxs = randInt(0, this.codeDim-1, this.batchSize);
                const gCode = tf.oneHot(idxs, this.codeDim);

                const fakeSamples = this.generator.predict([gLatent, gCode]);
                const fakeSamplesVal = fakeSamples.arraySync();

                // Store values in the buffer for plotting later
                for (let i = 0; i < this.batchSize; i++) {
                    this.codeSamplesBuff.set(idxs[i], i);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][0], i, 0);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][1], i, 1);
                }

                const dInputs = tf.concat([realSamples, fakeSamples]);

                const logValues = { iter, gLoss: 0, dLoss: 0, qLoss: 0 };

                // Train Discriminator
                this.discriminator.trainable = true;
                this.generator.trainable = false;
                this.qNetwork.trainable = false;
                const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);
                logValues.dLoss = dLoss;

                // Train Generator and Q Network
                this.discriminator.trainable = false;
                this.generator.trainable = true;
                this.qNetwork.trainable = false;
                const gqLoss = await this.combinedModel.trainOnBatch([gLatent, gCode], [realLabels, gCode]);
                logValues.gLoss = gqLoss[1];
                logValues.qLoss = gqLoss[2];

                tf.dispose([
                    realSamples, 
                    gLatent,
                    gCode,
                    fakeSamples,
                    dInputs,
                ])
                if (callback) await callback(logValues);
                iter++;
            }
            tf.dispose([realLabels, fakeLabels, dLabels]);
            this.isTraining = false;
        }


        dispose() {
            tf.dispose([this.generator, this.discriminator, this.qNetwork, this.combinedModel]);
        }
    }
    
    class ModelHandler {
        constructor(inputData) {
            this.inputData = inputData;
            this.gan = new InfoGAN();
            this.isInitialized = false;
            
            // Build the DOM entry
            this.modelEntry = d3.select('#modelEntryContainer').append('div')
                .attr('class', 'modelEntry')
                .attr('id', 'InfoGAN');

            this.modelEntry
            const modelCard = this.modelEntry.append('div')
                .classed('wrapContainer', true);

            const description = modelCard.append('div')
                .classed('wrappedItem', true);
            
            InfoGANDiagram.constructDescription(description);
                
            const diagram = modelCard.append('div')
                .attr('id', 'InfoGANDiagram')
                .classed('wrappedItem', true);

            InfoGANDiagram.constructDiagram(diagram); 
                
            this.trainToggleButton = description.append('button')
                .text('Train')
                .on('click', () => this.trainToggle());

            description.append('button')
                .text('Reset')
                .on('click', () => this.reset());

            // make the plots appear side by side
            this.QNetworkPlot = modelCard.append('div')
                .classed('wrappedItem', true)
                .classed('plot', true)
                .append('svg');

            
            this.discriminatorPlot = modelCard.append('div')
                .classed('wrappedItem', true)
                .classed('plot', true)
                .append('svg');

            
        }

        async init() {
            // change the button text
            this.trainToggleButton.text('Initializing...');
            this.gridshape = [15, 15];
            const x = tf.linspace(-1, 1, this.gridshape[0]);
            const y = tf.linspace(-1, 1, this.gridshape[1]);
            const grid = tf.meshgrid(x, y);
            this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
            tf.dispose([x, y, grid]);

            this.QNetworkPlot = new DynamicMultiDecisionMap({
                group: this.QNetworkPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                numMaps: 3,
                gridShape: this.gridshape,
            });

            this.discriminatorPlot = new DynamicDecisionMap({
                group: this.discriminatorPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                gridShape: this.gridshape,
            });

            await this.gan.init();

            this.fpsCounter = new FPSCounter("InfoGAN FPS");
            this.gLossVisor = new VisLogger({
                name: 'Generator Loss',
                tab: 'InfoGAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });
            this.dLossVisor = new VisLogger({
                name: 'Discriminator Loss',
                tab: 'InfoGAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });
            this.qLossVisor = new VisLogger({
                name: 'Q Network Loss',
                tab: 'InfoGAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });
            

            this.callback = async ({iter, gLoss, dLoss, qLoss}) => {
                this.gLossVisor.push({ x: iter, y: gLoss });
                this.dLossVisor.push({ x: iter, y: dLoss });
                this.qLossVisor.push({ x: iter, y: qLoss });
                
                // const realData = d3.shuffle(this.inputData).slice(0, 20);
                // const fakeData = this.generate(20);
                // Use the realDataBuff and fakeDataBuff to read data from
                
                const {realData, codeData, fakeData} = this.gan.readTrainingBuffer();
                const {decisionMap: ddm, gradientMap: dgm} = this.DiscriminatorDAGMap();
                this.discriminatorPlot.update({
                    realData,
                    codeData,
                    fakeData,
                    decisionMap: ddm,
                    gradientMap: dgm
                })

                const {decisionMaps: qdm, gradientMap: qgm} = this.QDAGMaps();
                this.QNetworkPlot.update({
                    realData, 
                    codeData, 
                    fakeData,
                    decisionMaps: qdm, 
                    gradientMap: qgm
                });

                this.fpsCounter.update();
            }

            this.isInitialized = true;
        }

        generate(nSamples) {
            return this.gan.generate(nSamples);
        }

        // DAG = Decision and Gradient
        DiscriminatorDAGMap() {
            const points = this.decisionMapInputBuff;
            const {value: pred, grad} = tf.valueAndGrad(
                point => {
                    return this.gan.discriminator.predict(point);
                })(points);

            const xyuv = tf.concat([points, grad], 1);
            const ret = {
                decisionMap: pred.arraySync(),
                gradientMap: xyuv.arraySync()
            };

            tf.dispose([pred, grad, xyuv]);
            return ret;
        }

        QDAGMaps() {
            const points = this.decisionMapInputBuff;
            const preds = this.gan.qNetwork.predict(points);
            const predsT = preds.transpose();

            // Only return the highest probability's gradient
            const grad = tf.grad((point) => {
                return this.gan.qNetwork.predict(point).max(1);
            })(points);

            const xyuv = tf.concat([points, grad], 1);

            // To generate the gradients for each output:
            // const preds = [];
            // const xyuvs = [];
            // for (let i = 0; i < this.gan.codeDim; i++) {
            //     const {value: pred, grad} = tf.valueAndGrad((point) => {
            //         return this.gan.qNetwork.predict(point).max({ axis: 1});
            //     })(points);

            //     preds.push(pred.arraySync());

            //     const xyuv = tf.concat([points, grad], 1);
            //     xyuvs.push(xyuv.arraySync());

            //     tf.dispose([pred, grad, xyuv]);
            // }

            const ret = {
                decisionMaps: predsT.arraySync(),
                gradientMap: xyuv.arraySync()
            };
            tf.dispose([preds, predsT, grad, xyuv]);
            return ret;
        }

        async trainToggle() {
            if (!this.isInitialized) await this.init();
            this.gan.trainToggle(this.inputData, this.callback);
            // change the button text
            if (this.gan.isTraining) {
                this.trainToggleButton.text('Stop Training');
            } else {
                this.trainToggleButton.text('Resume Training');
            }
        }

        async reset() {
            this.gan.resetParams();
            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.qLossVisor.clear();
        }
    }

    return { ModelHandler };
})();
