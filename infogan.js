// InfoGAN.js

const InfoGAN = (function() {
    // Function to generate fake samples
    function generateFakeSamples(generator, latentDim, codeDim, nSamples) {
        const latentPoints = tf.randomNormal([nSamples, latentDim]);
        // Random one-hot codes
        const idxs = randInt(0, codeDim, nSamples);
        const codes = tf.oneHot(idxs, codeDim);
        const input = tf.concat([latentPoints, codes], 1);
        const samples = generator.predict(input);
        tf.dispose([latentPoints, codes, input]);
        return samples;
    }

    // InfoGAN class
    class InfoGAN {
        constructor(
            {
                latentDim = 100,
                codeDim = 10,
                genLayers = 4,
                genStartDim = 128,
                discLayers = 4,
                discStartDim = 512,
                batchSize = 1024,
                qWeight = 0.1,
            } = {}
        ) {
            this.generator = null;
            this.discriminator = null;
            this.qNetwork = null;
            this.latentDim = latentDim;
            this.codeDim = codeDim;
            this.genLayers = genLayers;
            this.genStartDim = genStartDim;
            this.discLayers = discLayers;
            this.discStartDim = discStartDim;
            this.batchSize = batchSize;
            this.qWeight = qWeight;
        }

        async init() {
            this.buildGenerator(this.latentDim, this.codeDim, this.genLayers, this.genStartDim);
            this.buildBaseNetwork(this.discLayers, this.discStartDim);
            this.buildQNetwork(this.codeDim);
            this.buildDiscriminator(this.discLayers, this.discStartDim, this.codeDim);
            this.buildDQLoss();
            // this.gan = tf.model({
            //     inputs: this.gInput,
            //     outputs: this.jointLoss,
            // });
            // this.gan.compile({
            //     optimizer: tf.train.adam(0.0001),
            //     // optimizer: tf.train.sgd(0.0001),
            //     // loss: 'binaryCrossentropy',
            // });

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

            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.isTraining = false;
        }

        buildGenerator(latentDim, codeDim, numLayers = 4, startDim = 128) {
            this.gInput = tf.input({ shape: [this.latentDim + this.codeDim] });
            this.generator = tf.sequential();
            const totalDim = latentDim + codeDim;
            generator.add(tf.layers.dense({ units: startDim, inputShape: [totalDim], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
            }

            generator.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Output layer for 2D points
            this.gOutput = generator.apply(this.gInput);
        }

        // Common base network for Discriminator and Q Network
        buildBaseNetwork(numLayers = 4, startDim = 512) {
            this.baseNetwork = tf.sequential();
            this.baseNetwork.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));

            for (let i = 1; i < numLayers; i++) {
                this.baseNetwork.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }
            this.baseOutput = baseNetwork.apply(this.gOutput);
        }

        buildQNetwork(codeDim) {
            const inputShape = this.baseOutput.shape.slice(1);
            this.qNetwork = tf.sequential();
            this.qNetwork.add(tf.layers.dense({ units: codeDim, inputShape: inputShape, activation: 'softmax' }));
            this.qNetworkOutput = qNetwork.apply(this.baseOutput);
        }

        buildDiscriminator(numLayers = 4, startDim = 512, codeDim) {
            // Discriminator output layer
            const inputShape = this.baseOutput.shape.slice(1);
            this.discriminator = tf.sequential();
            this.discriminator.add(tf.layers.dense({ units: 1, inputShape: inputShape, activation: 'sigmoid' }));
            this.discriminatorOutput = this.discriminator.apply(this.baseOutput);
        }

        buildDQLoss() {
            this.dLabels = tf.input({ shape: [1] });
            this.qLabels = tf.input({ shape: [this.codeDim] });

            // Compute the loss for the discriminator
            this.dLoss = tf.losses.binaryCrossentropy(
                this.discriminatorOutput, this.dLabels
            );
            this.qLoss = tf.losses.categoricalCrossentropy(
                this.qNetworkOutput, this.qLabels
            );

            // Combine the losses
            this.jointLoss = tf.tidy(() => {
                const qLoss = tf.scalar(this.qWeight).mul(this.qLoss);
                return this.dLoss.add(qLoss);
            });
        }

        async resetParams() {
            await Promise.all([
                resetWeights(this.generator),
                resetWeights(this.discriminator),
                resetWeights(this.qNetwork)
            ]);

            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.qLossVisor.clear();
        }

        async trainToggle(data, callback = null) {
            if (this.isTraining) {
                this.isTraining = false;
                return;
            }

            this.isTraining = true;
            let iter = 0;
            const numEpochs = 10000; // Add stopping criterion like epochs

            const realLabels = tf.ones([this.batchSize, 1]);
            const fakeLabels = tf.zeros([this.batchSize, 1]);
            const dLabels = tf.concat([realLabels, fakeLabels]);

            while (this.isTraining && iter < numEpochs) {
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const realSamples = this.realSamplesBuff.toTensor();
                const fakeSamples = generateFakeSamples(this.generator, this.latentDim, this.codeDim, this.batchSize);


                this.discriminator.trainable = true;
                // const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);


                const latentPoints = tf.randomNormal([this.batchSize, this.latentDim]);
                const codeInts = randInt(0, this.codeDim, this.batchSize);
                const codeOH = tf.oneHot(codeInts, this.codeDim);
                const ganInput = tf.concat([latentPoints, codeOH], 1);
                const misleadingLabels = tf.ones([this.batchSize, 1]);

                this.discriminator.trainable = false;
                const dInputs = tf.concat([realSamples, fakeSamples]);
                const gLoss = await this.gan.trainOnBatch(ganInput, misleadingLabels);

                const predictedCodes = this.qNetwork.predict(fakeSamples);
                const qLoss = await this.qNetwork.trainOnBatch(fakeSamples, codeOH);

                if (callback) callback(iter, gLoss, dLoss, qLoss);

                tf.dispose([
                    realSamples,
                    fakeSamples,
                    dInputs,
                    latentPoints,
                    codeInts,
                    codeOH,
                    ganInput,
                    misleadingLabels,
                    predictedCodes
                ]);

                iter++;
            }
            this.isTraining = false;
            tf.dispose([realLabels, fakeLabels, dLabels]);
        }


        generate(nSamples) {
            return tf.tidy(() => {
                const latentPoints = tf.randomNormal([nSamples, this.latentDim]);
                const codeInts = Array.from(
                    { length: nSamples },
                    () => Math.floor(Math.random() * this.codeDim)
                );
                const codeOH = tf.oneHot(codeInts, this.codeDim);
                const input = tf.concat([latentPoints, codeOH], 1);
                const pred = this.generator.predict(input);
                const ret = pred.arraySync();
                return ret;
            });
        }

        dispose() {
            tf.dispose([this.generator, this.discriminator, this.qNetwork, this.gan]);
        }
    }


    class ModelHandler {
        constructor(inputData) {
            this.inputData = inputData;
            this.gan = new InfoGAN();
            this.ddm = new DynamicDecisionMap({
                div: '#mainGANPlot',
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
            });

            this.fpsCounter = new FPSCounter("InfoGAN FPS");
            this.isInitialized = false;
        }

        async init() {
            await this.gan.init();
            this.callback = (iter, gLoss, dLoss, qLoss) => {
                this.gan.gLossVisor.push({ x: iter, y: gLoss });
                this.gan.dLossVisor.push({ x: iter, y: dLoss });
                this.gan.qLossVisor.push({ x: iter, y: qLoss });
                this.ddm.plot(this);
            };

            this.gridSize = 20;
            const x = tf.linspace(-1, 1, this.gridSize);
            const y = tf.linspace(-1, 1, this.gridSize);
            const grid = tf.meshgrid(x, y);
            this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
            tf.dispose([x, y, grid]);

            this.isInitialized = true;
        }

        generate(nSamples) {
            return this.gan.generate(nSamples);
        }

        decisionAndGradientMap() {
            return tf.tidy(() => {
                const points = this.decisionMapInputBuff;
                const res = tf.valueAndGrad(point => this.gan.discriminator.predict(point))(points);

                const pred2D = res.value.reshape([this.gridSize, this.gridSize]);
                const xyuv = tf.concat([points, res.grad], 1);
                const xyuv2D = xyuv.reshape([this.gridSize, this.gridSize, 4]);
                const ret = {
                    decisionMap: pred2D.arraySync(),
                    gradientMap: xyuv2D.arraySync()
                };
                return ret;
            });
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
            this.ddm.plot(this);
        }
    }

    return { ModelHandler };
})();
