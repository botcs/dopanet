// gan.js

const VanillaGAN = (function() {
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
    

    class VanillaGAN {
        constructor(
            {
                latentDim = 100,
                genLayers = 1,
                genStartDim = 1024,
                discLayers = 1,
                discStartDim = 1024,
                batchSize = 64
            } = {}
        ) {
            this.generator = null;
            this.discriminator = null;
            this.gan = null;
            this.latentDim = latentDim;
            this.genLayers = genLayers;
            this.genStartDim = genStartDim;
            this.discLayers = discLayers;
            this.discStartDim = discStartDim;
            this.batchSize = batchSize;
        }

        async init() {
            this.buildGenerator({
                latentDim: this.latentDim, 
                genLayers: this.genLayers, 
                genStartDim: this.genStartDim
            });
            this.buildDiscriminator(this.discLayers, this.discStartDim);
            this.buildCombinedModel();
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.fakeSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.isTraining = false;
        }

        buildGenerator({latentDim, numLayers = 4, startDim = 128}) {
            this.gLatent = tf.input({ shape: [latentDim] });
            
            const backbone = tf.sequential();
            backbone.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                backbone.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
            }
    
            backbone.add(tf.layers.dense({ units: 2, activation: 'linear' }));
            const gOutput = backbone.apply(this.gLatent);

            this.generator = tf.model({ inputs: this.gLatent, outputs: gOutput });
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


        buildCombinedModel() {
            const gOutput = this.generator.outputs[0];
            const dOutput = this.discriminator.apply(gOutput);

            this.combinedModel = tf.model({ inputs: this.generator.inputs, outputs: dOutput });
            const gLossFn = (yTrue, yPred) => tf.metrics.binaryCrossentropy(yTrue, yPred).mul(tf.scalar(1));
            this.combinedModel.compile({ 
                optimizer: tf.train.adam(0.0001, 0.5, 0.5), 
                // optimizer: tf.train.momentum(0.001, 0.1),
                loss: gLossFn,
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

            const realLabels = tf.ones([this.batchSize, 1]);
            const fakeLabels = tf.zeros([this.batchSize, 1]);
            const dLabels = tf.concat([realLabels, fakeLabels]);

            while (this.isTraining) {
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)]
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const realSamples = this.realSamplesBuff.toTensor();
                const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
                const fakeSamples = this.generator.predict(gLatent);

                // Store values in the buffer for plotting later
                const fakeSamplesVal = fakeSamples.arraySync();
                for (let i = 0; i < this.batchSize; i++) {
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][0], i, 0);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][1], i, 1);
                }

                const dInputs = tf.concat([realSamples, fakeSamples]);
                const logValues = { iter, gLoss: 0, dLoss: 0};

                // Train Discriminator
                this.discriminator.trainable = true;
                this.generator.trainable = false;
                const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);
                logValues.dLoss = dLoss;

                // Train Generator and Q Network
                this.discriminator.trainable = false;
                this.generator.trainable = true;
                logValues.gLoss = await this.combinedModel.trainOnBatch(gLatent, realLabels);
                 
                tf.dispose([
                    realSamples, 
                    gLatent,
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
            tf.dispose([this.generator, this.discriminator, this.gan]);
        }
    }

    class ModelHandler {
        constructor(inputData) {
            this.inputData = inputData;
            this.gan = new VanillaGAN();
            this.isInitialized = false;

            // Build the DOM entry
            this.modelEntry = d3.select('#modelEntryContainer')
                .append('div')
                .attr('class', 'modelEntry')
                .attr('id', 'VanillaGAN');

            this.modelEntry
            const modelCard = this.modelEntry.append('div')
                .classed('wrapContainer', true);

            const description = modelCard.append('div')
                .classed('wrappedItem', true);
            
            VanillaGANDiagram.constructDescription(description);
                
            const diagram = modelCard.append('div')
                .attr('id', 'InfoGANDiagram')
                .classed('wrappedItem', true);

            VanillaGANDiagram.constructDiagram(diagram); 

            const explanation = modelCard.append('div')
                .classed('wrappedItem', true);
            VanillaGANDiagram.constructExplanation(explanation);


            this.trainToggleButton = explanation.append('button')
            .text('Train')
            .on('click', () => this.trainToggle());

            explanation.append('button')
                .text('Reset')
                .on('click', () => this.reset());



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

            this.discriminatorPlot = new DynamicDecisionMap({
                group: this.discriminatorPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                gridShape: this.gridshape,
                showColorbar: true,
            });

            await this.gan.init();

            this.fpsCounter = new FPSCounter("Vanilla GAN FPS");
            this.gLossVisor = new VisLogger({
                name: 'Generator Loss',
                tab: 'Vanilla GAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });
            this.dLossVisor = new VisLogger({
                name: 'Discriminator Loss',
                tab: 'Vanilla GAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });
            

            this.callback = async ({iter, gLoss, dLoss}) => {
                this.gLossVisor.push({ x: iter, y: gLoss });
                this.dLossVisor.push({ x: iter, y: dLoss });
                
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

                this.fpsCounter.update();
            }

            this.isInitialized = true;
        }

        generate(nSamples) { 
            return this.gan.generate(nSamples) 
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
            this.gan.reset();
            this.gLossVisor.clear();
            this.dLossVisor.clear();
        }
    }
    return { ModelHandler };
})();