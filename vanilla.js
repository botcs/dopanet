// gan.js

const VanillaGAN = (function() {
    // Generator Model
    function buildGenerator(latentDim, numLayers = 4, startDim = 128) {
        const generator = tf.sequential();
        generator.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'relu' }));

        for (let i = 1; i < numLayers; i++) {
            // add batch normalization
            // generator.add(tf.layers.layerNormalization());
            generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu'}));
        }

        // generator.add(tf.layers.layerNormalization());
        generator.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Output layer for 2D points
        return generator;
    }

    // Discriminator Model
    function buildDiscriminator(numLayers = 4, startDim = 512) {
        const discriminator = tf.sequential();
        discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));

        for (let i = 1; i < numLayers; i++) {
            // discriminator.add(tf.layers.layerNormalization());
            discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu'}));
        }

        discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Output layer
        discriminator.compile({
            optimizer: tf.train.adam(0.0005),
            // optimizer: tf.train.sgd(0.001),
            loss: 'binaryCrossentropy',
        });
        return discriminator;
    }

    // Function to generate fake samples
    function generateFakeSamples(generator, latentDim, nSamples) {
        const latentPoints = tf.randomNormal([nSamples, latentDim]);
        const samples = generator.predict(latentPoints);
        tf.dispose([latentPoints]);
        return samples;
    }


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
            this.generator = buildGenerator(this.latentDim, this.genLayers, this.genStartDim);
            this.discriminator = buildDiscriminator(this.discLayers, this.discStartDim);
            this.gan = tf.sequential();
            this.gan.add(this.generator);
            this.gan.add(this.discriminator);
            this.gan.compile({
                // optimizer: tf.train.sgd(0.001),
                optimizer: tf.train.adam(0.0001),
                loss: 'binaryCrossentropy',
            });

            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);

            this.isTraining = false;
        }

        async resetParams() {
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
                const fakeSamples = generateFakeSamples(this.generator, this.latentDim, this.batchSize);

                const dInputs = tf.concat([realSamples, fakeSamples]);

                this.discriminator.trainable = true;
                const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);

                const latentPoints = tf.randomNormal([this.batchSize, this.latentDim]);

                this.discriminator.trainable = false;
                const gLoss = await this.gan.trainOnBatch(latentPoints, realLabels);

                if (callback) callback(iter, gLoss, dLoss);

                tf.dispose([
                    realSamples, 
                    fakeSamples,
                    dInputs,
                    latentPoints
                ]);
                iter++;
            }
            this.isTraining = false;
            tf.dispose([realLabels, fakeLabels, dLabels]);
        }

        generate(nSamples) {
            const latentPoints = tf.randomNormal([nSamples, this.latentDim]);
            const pred = this.generator.predict(latentPoints);
            const ret = pred.arraySync();
            tf.dispose([latentPoints, pred]);
            return ret;
        }

        dispose() {
            tf.dispose([this.generator, this.discriminator, this.gan]);
        }
    }

    class ModelHandler {
        constructor(inputData) {
            this.inputData = inputData;
            this.gan = new VanillaGAN();
            this.ddm = new DynamicDecisionMap({
                div: '#mainGANPlot',
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
            });

            this.isInitialized = false;
        }
        async init() {
            await this.gan.init();

            this.gridSize = 20;
            const x = tf.linspace(-1, 1, this.gridSize);
            const y = tf.linspace(-1, 1, this.gridSize);
            const grid = tf.meshgrid(x, y);
            this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
            tf.dispose([x, y, grid]);


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

            this.callback = (iter, gLoss, dLoss) => {
                this.gLossVisor.push({ x: iter, y: gLoss });
                this.dLossVisor.push({ x: iter, y: dLoss });
                this.ddm.plot(this);
                this.fpsCounter.update();
            }
            
            this.isInitialized = true;
        }

        generate(nSamples) { return this.gan.generate(nSamples) }


        decisionAndGradientMap() {
            const points = this.decisionMapInputBuff;

            // Use tf.valueAndGrad to get both predictions and gradients
            const res = tf.valueAndGrad(point => this.gan.discriminator.predict(point))(points);
            
            const pred2D = res.value.reshape([this.gridSize, this.gridSize]);

            const xyuv = tf.concat([points, res.grad], 1);
            const xyuv2D = xyuv.reshape([this.gridSize, this.gridSize, 4]);
            const ret = { 
                decisionMap: pred2D.arraySync(), 
                gradientMap: xyuv2D.arraySync() 
            };
            tf.dispose([points, res, pred2D, xyuv, xyuv2D]);
            return ret;
        }

        async trainToggle(data) {
            if (!this.isInitialized) await this.init();
            this.gan.trainToggle(data, this.callback);
        }

        async stopTraining() {
            this.gan.isTraining = false;
        }

        async reset() {
            this.gan.resetParams();
            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.ddm.plot(this);
        }
    }
    return { ModelHandler };
})();