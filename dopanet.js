// dopanet.js
const DoPaNet = (function() {
    // Function to reset the weights
    async function resetWeights(model, initializerName = 'glorotNormal') {
        for (let layer of model.layers) {
            if (layer.getWeights().length > 0) {
                // Get the shape of the weights
                const originalWeights = layer.getWeights();
                const resetWeights = originalWeights.map(weight => {
                    const shape = weight.shape;
                    return tf.initializers[initializerName]().apply(shape);
                });

                // Set the weights
                layer.setWeights(resetWeights);
            }
        }
    }

    class DoPaNet {
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
                gLR = 0.0001,
                dLR = 0.0001,
                qLR = 0.0001
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
            this.gLR = gLR;
            this.dLR = dLR;
            this.qLR = qLR;

        }
    
        async init() {
            this.buildGenerators({
                latentDim: this.latentDim, 
                codeDim: this.codeDim, 
                numLayers: this.genLayers, 
                startDim: this.genStartDim
            });
            this.buildDiscriminators({
                numLayers: this.discLayers, 
                startDim: this.discStartDim
            });
            this.buildQNetwork({
                codeDim: this.codeDim, 
                numLayers: this.discLayers, 
                startDim: this.discStartDim
            });
            this.buildCombinedModels();
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.codeSamplesBuff = tf.buffer([this.batchSize*this.codeDim]);
            this.fakeSamplesBuff = tf.buffer([this.batchSize*this.codeDim, 2]);
            this.isTraining = false;
        }
    
        
        buildGenerators({latentDim, numLayers = 4, startDim = 128}) {
            this.generators = [];
            for (let i = 0; i < this.codeDim; i++) {
                const generator = tf.sequential();
                generator.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'relu' }));
                for (let i = 1; i < numLayers; i++) {
                    generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
                }
                generator.add(tf.layers.dense({ units: 2, activation: 'linear' }));
                this.generators.push(generator);
            }
        }

    
        buildDiscriminators({numLayers, startDim}={}) {
            this.discriminators = [];
            for (let i = 0; i < this.codeDim; i++) {
                const discriminator = tf.sequential();
                discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
                for (let i = 1; i < numLayers; i++) {
                    discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
                }
                discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
                discriminator.compile({
                    optimizer: tf.train.adam(this.dLR),
                    loss: 'binaryCrossentropy',
                });
                this.discriminators.push(discriminator);
            }
        }
    
        buildQNetwork({codeDim, numLayers, startDim}) {
            const qNetwork = tf.sequential();
            qNetwork.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
            for (let i = 1; i < numLayers; i++) {
                qNetwork.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }

            qNetwork.add(tf.layers.dense({ units: codeDim, activation: 'softmax' }));
            
            qNetwork.compile({
                optimizer: tf.train.adam(this.qLR),
                loss: 'categoricalCrossentropy',
            });
            this.qNetwork = qNetwork;
        }

        buildCombinedModels() {
            this.combinedModels = [];
            for (let i = 0; i < this.codeDim; i++) {
                // const combinedModel = tf.sequential();
                // combinedModel.add(this.generators[i]);
                // this.discriminators[i].trainable = false;
                // combinedModel.add(this.discriminators[i]);

                const gInput = this.generators[i].input;
                const gOutput = this.generators[i].output;

                // Honestly, tfjs made it impossible to know when to turn
                // the trainable attribute off and on
                this.discriminators[i].trainable = false;
                const dOutput = this.discriminators[i].apply(gOutput);
                const qOutput = this.qNetwork.apply(gOutput);
                const combinedModel = tf.model({
                    inputs: gInput,
                    outputs: [dOutput, qOutput]
                });
                const gLossFn = (yTrue, yPred) => tf.metrics.binaryCrossentropy(yTrue, yPred);
                const qLossFn = (yTrue, yPred) => tf.metrics.sparseCategoricalCrossentropy(yTrue, yPred).mul(tf.scalar(this.qWeight));
                combinedModel.compile({
                    optimizer: tf.train.adam(this.gLR),
                    loss: [gLossFn, qLossFn],
                });
                this.combinedModels.push(combinedModel);
            }
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

        async reset() {
            // reset the weights and biases of the generator and discriminator
            
            // await together for both models
            await Promise.all([
                resetWeights(this.generator), 
                resetWeights(this.discriminator)
            ]);
            
        }

        async trainToggle(data, callback = null) {
            if (this.isTraining) {
                this.isTraining = false;
                return;
            }
    
            this.isTraining = true;
            let iter = 0;
            const realLabels = tf.fill([this.batchSize], 1);
            const fakeLabels = tf.fill([this.batchSize], 0);
            const qLabelsJS = [];
            for (let i = 0; i < this.codeDim; i++) {
                qLabelsJS.push(Array(this.batchSize).fill(i));
            }
            const qLabels = tf.tensor(qLabelsJS.flat());
            // save to codeSamplesBuff
            for (let i = 0; i < this.batchSize * this.codeDim; i++) {
                this.codeSamplesBuff.set(qLabelsJS[i], i);
            }

            while (this.isTraining) { 
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const realSamples = this.realSamplesBuff.toTensor();
                const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
                const fakeSamplesList = this.generators.map(gen => gen.predictOnBatch(gLatent));
                const fakeSamples = tf.concat(fakeSamplesList);
                const fakeSamplesVal = fakeSamples.arraySync();
                
                // Store values in the buffer for plotting later
                for (let i = 0; i < this.batchSize; i++) {
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][0], i, 0);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][1], i, 1);
                }

                const logValues = { iter };


                // Train Q Network
                // this.generators.forEach(gen => gen.trainable = false);
                // this.discriminators.forEach(disc => disc.trainable = false);
                // this.qNetwork.trainable = true;
                // const qLoss = await this.qNetwork.trainOnBatch(fakeSamples, qLabels);


                // Train Discriminator
                this.discriminators.forEach(disc => disc.trainable = true);
                this.generators.forEach(gen => gen.trainable = false);
                // evaluate the QNetwork on the real samples
                // and gather them for the corresponding discriminators
                const qRealPred = this.qNetwork.predictOnBatch(realSamples);
                const qRealPredArgmax = tf.argMax(qRealPred, 1);
                const dTrainingPromises = [];
                for (let i = 0; i < this.codeDim; i++) {
                    // gather the real samples for the corresponding discriminator
                    const mask = tf.equal(qRealPredArgmax, i);
                    const dInput = await tf.booleanMaskAsync(realSamples, mask);
                    // CSABI: START FROM HERE TOMORROW
                    const discriminator = this.discriminators[i];
                    const promise = discriminator.trainOnBatch(dInput, dLabels);
                    dTrainingPromises.push(promise);
                    tf.dispose(mask, realSamplesForDisc);
                }
                const dLosses = await Promise.all(dTrainingPromises);

                logValues.dLosses = dLosses;
                logValues.avgDLoss = dLosses.reduce((a, b) => a + b) / dLosses.length;


                // Train Generator and Q Network
                this.discriminators.forEach(disc => disc.trainable = false);
                this.generators.forEach(gen => gen.trainable = true);
                const gqTrainingPromises = [];
                for (let i = 0; i < this.codeDim; i++) {
                    const combinedModel = this.combinedModel[i];
                    const promise = combinedModel.trainOnBatch(gLatent, realLabels);
                    gqTrainingPromises.push(promise);
                }
                const gqLosses = await Promise.all(gqTrainingPromises);
                logValues.gLosses = gqLosses.map(losses => losses[1]);
                logValues.avgGLoss = gqLosses.reduce((a, b) => a + b[1]) / gqLosses.length;
                logValues.qLoss = gqLosses.reduce((a, b) => a + b[2]) / gqLosses.length;

                tf.dispose([
                    realSamples, 
                    gLatent,
                    fakeSamples,
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
            this.gan = new DoPaNet();
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
            
            DoPaNetDiagram.constructDescription(description);
                
            const diagram = modelCard.append('div')
                .attr('id', 'DoPaNetDiagram')
                .classed('wrappedItem', true);

            DoPaNetDiagram.constructDiagram(diagram); 
                
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

            // this.QNetworkPlot = new DynamicMultiDecisionMap({
            //     group: this.QNetworkPlot,
            //     xlim: [-1, 1],
            //     ylim: [-1, 1],
            //     zlim: [0, 1],
            //     numMaps: 3,
            //     gridShape: this.gridshape,
            // });

            // this.discriminatorPlot = new DynamicDecisionMap({
            //     group: this.discriminatorPlot,
            //     xlim: [-1, 1],
            //     ylim: [-1, 1],
            //     zlim: [0, 1],
            //     gridShape: this.gridshape,
            // });

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
                
                // const {realData, codeData, fakeData} = this.gan.readTrainingBuffer();
                // const {decisionMap: ddm, gradientMap: dgm} = this.DiscriminatorDAGMap();
                // this.discriminatorPlot.update({
                //     realData,
                //     codeData,
                //     fakeData,
                //     decisionMap: ddm,
                //     gradientMap: dgm
                // })

                // const {decisionMaps: qdm, gradientMap: qgm} = this.QDAGMaps();
                // this.QNetworkPlot.update({
                //     realData, 
                //     codeData, 
                //     fakeData,
                //     decisionMaps: qdm, 
                //     gradientMap: qgm
                // });

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
                    return this.gan.discriminator.predictOnBatch(point);
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
            const preds = this.gan.qNetwork.predictOnBatch(points);
            const predsT = preds.transpose();

            // Only return the highest probability's gradient
            const grad = tf.grad((point) => {
                return this.gan.qNetwork.predictOnBatch(point).max(1);
            })(points);

            const xyuv = tf.concat([points, grad], 1);

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
            
            tfvis.visor().setActiveTab('InfoGAN');

            // change the button text
            if (this.gan.isTraining) {
                this.trainToggleButton.text('Stop Training');
            } else {
                this.trainToggleButton.text('Resume Training');
            }
        }

        async reset() {
            this.gan.reset();
            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.qLossVisor.clear();
        }
    }

    return { ModelHandler };
})();
