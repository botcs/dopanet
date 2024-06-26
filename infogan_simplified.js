// InfoGAN.js

const InfoGAN = (function() {
    class InfoGAN {
        constructor(
            {
                latentDim = 100,
                codeDim = 3,
                genLayers = 1,
                genStartDim = 1024,
                discLayers = 1,
                discStartDim = 1024,
                batchSize = 64,
                qWeight = 1,
                latentNorm = 1/2,
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
                genLayers: this.genLayers, 
                genStartDim: this.genStartDim
            });
            this.buildDiscriminator(this.discLayers, this.discStartDim);
            this.buildQNetwork(this.codeDim, this.discLayers, this.discStartDim);
            this.buildCombinedModel();
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.fakeSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.isTraining = false;
        }
    
        buildGenerator({latentDim, codeDim, numLayers = 4, startDim = 128}) {
            this.gLatent = tf.input({ shape: [latentDim] });
            this.gCode = tf.input({ shape: [codeDim] });
            
            const latentEmb = tf.layers.dense({ units: startDim, activation: 'linear', useBias: false }).apply(this.gLatent);
            const codeEmb = tf.layers.dense({ units: startDim, activation: 'linear', useBias: false }).apply(this.gCode);

            // const gEmbedding = tf.layers.add().apply([latentEmb, codeEmb]);
            const normLatent = new MultiplyLayer({ constant: this.latentNorm }).apply(latentEmb);
            const gEmbedding = tf.layers.add().apply([normLatent, codeEmb]);
            
            const backbone = tf.sequential();
            backbone.add(tf.layers.dense({ units: startDim, inputShape: [startDim], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                backbone.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
            }
    
            backbone.add(tf.layers.dense({ units: 2, activation: 'linear' }));
            const gOutput = backbone.apply(gEmbedding);

            this.generator = tf.model({ inputs: [this.gLatent, this.gCode], outputs: gOutput });
        }
    
        buildDiscriminator(numLayers = 4, startDim = 512) {
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
    
        buildQNetwork(codeDim, numLayers = 4, startDim = 512) {
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
            const fakeData = this.fakeSamplesBuff.toTensor();

            const ret = {
                realData: realData.arraySync(),
                fakeData: fakeData.arraySync(),
            }
            tf.dispose([realData, fakeData]);
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
                const idxs = randInt(0, this.codeDim, this.batchSize);
                const gCode = tf.oneHot(idxs, this.codeDim);

                const fakeSamples = this.generator.predict([gLatent, gCode]);
                const fakeSamplesVal = fakeSamples.arraySync();
                // load it to the FakeSamples buffer
                for (let i = 0; i < this.batchSize; i++) {
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][0], i, 0);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][1], i, 1);
                }

                const dInputs = tf.concat([realSamples, fakeSamples]);
                const dLabelsCombined = tf.concat([realLabels, fakeLabels]);

                const logValues = { iter, gLoss: 0, dLoss: 0, qLoss: 0 };

                // Train Discriminator
                this.discriminator.trainable = true;
                this.generator.trainable = false;
                this.qNetwork.trainable = false;
                const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabelsCombined);
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
                    dLabelsCombined,
                ])
                if (callback) await callback(logValues);
                iter++;
            }
            tf.dispose([realLabels, fakeLabels, dLabels]);
            this.isTraining = false;
        }

        generate(nSamples) {
            const gLatent = tf.randomNormal([nSamples, this.latentDim]);
            const codeInts = Array.from(
                { length: nSamples },
                () => Math.floor(Math.random() * this.codeDim)
            );
            const gCode = tf.oneHot(codeInts, this.codeDim);
            const pred = this.generator.predict([gLatent, gCode]);
            const ret = pred.arraySync();
            tf.dispose([gLatent, gCode, pred]);
            return ret;   
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
            
            description.append('h3').text('InfoGAN')
            description.append('p')
                .text('InfoGAN is a GAN that incorporates a Q network to learn latent codes.Q network to learn latent codes.Q network to learn latent codes.Q network to learn latent codes.Q network to learn latent codes.Q network to learn latent codes.Q network to learn latent codes.')
                
                
            const diagram = modelCard.append('div')
                .attr('id', 'InfoGANDiagram')
                .classed('wrappedItem', true);

            InfoGANDiagram.constructDiagram(diagram);

            
            // modelEntry.append('p').text('InfoGAN is a GAN that incorporates a Q network to learn latent codes.');
            // const controlBox = modelCard.append('div')
            //     .classed('wrapContainer', true);
                
            this.trainToggleButton = description.append('button')
                .text('Train')
                .attr('id', 'trainInfoGANButton')
                .on('click', () => this.trainToggle());

            description.append('button')
                .text('Reset')
                .attr('id', 'resetInfoGANButton')
                .on('click', () => this.reset());

            // make the plots appear side by side


            this.discriminatorPlot = modelCard.append('div')
                // .attr('class', 'GANPlot')
                .classed('wrappedItem', true)
                .classed('plot', true)
                .append('svg');

            this.QNetworkPlot = modelCard.append('div')
                // .attr('class', 'GANPlot')
                .classed('wrappedItem', true)
                .classed('plot', true)
                .append('svg');

            
        }

        async init() {
            // change the button text
            this.trainToggleButton.text('Initializing...');

            this.QNetworkPlot = new DynamicMultiDecisionMap({
                group: this.QNetworkPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                numMaps: 3,
            });

            this.discriminatorPlot = new DynamicDecisionMap({
                group: this.discriminatorPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
            });

            await this.gan.init();
            
            this.callback = async ({iter, gLoss, dLoss, qLoss}) => {
                this.gLossVisor.push({ x: iter, y: gLoss });
                this.dLossVisor.push({ x: iter, y: dLoss });
                this.qLossVisor.push({ x: iter, y: qLoss });
                
                // const realData = d3.shuffle(this.inputData).slice(0, 20);
                // const fakeData = this.generate(20);
                // Use the realDataBuff and fakeDataBuff to read data from
                
                const {realData, fakeData} = this.gan.readTrainingBuffer();

                const {decisionMap: ddm, gradientMap: dgm} = this.DiscriminatorDAGMap();
                this.discriminatorPlot.update({
                    realData,
                    fakeData,
                    decisionMap: ddm,
                    gradientMap: dgm
                })

                const {decisionMaps: qdm, gradientMap: qgm} = this.QDAGMaps();
                this.QNetworkPlot.update({
                    realData, 
                    fakeData, 
                    decisionMaps: qdm, 
                    gradientMap: qgm
                });

                this.fpsCounter.update();
            }

            this.ddmGridSize = 20;
            const x = tf.linspace(-1, 1, this.ddmGridSize);
            const y = tf.linspace(-1, 1, this.ddmGridSize);
            const grid = tf.meshgrid(x, y);
            this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
            tf.dispose([x, y, grid]);

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

        async stopTraining() {
            this.gan.isTraining = false;
        }

        async reset() {
            this.gan.resetParams();

            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.qLossVisor.clear();
            this.ddm.plot(this);
        }
    }

    return { ModelHandler };
})();
