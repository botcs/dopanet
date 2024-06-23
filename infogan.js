// InfoGAN.js

const InfoGAN = (function() {
    class InfoGAN {
        constructor(
            {
                latentDim = 100,
                codeDim = 2,
                genLayers = 1,
                genStartDim = 32,
                discLayers = 1,
                discStartDim = 32,
                batchSize = 64,
                qWeight = 1,
                latentNorm = 2,
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
            this.latentNorm = latentNorm;
        }
    
        async init() {
            this.buildGenerator({
                latentDim: this.latentDim, 
                codeDim: this.codeDim, 
                genLayers:this.genLayers, 
                genStartDim: this.genStartDim
            });
            this.buildBaseNetwork(this.discLayers, this.discStartDim);
            this.buildQNetwork(this.codeDim);
            this.buildDiscriminator(this.discLayers, this.discStartDim, this.codeDim);
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.isTraining = false;
        }
    
        buildGenerator({latentDim, codeDim, numLayers = 4, startDim = 128}) {
            this.gLatent = tf.input({ shape: [latentDim] });
            this.gCode = tf.input({ shape: [codeDim] });
            
            this.latentEmbeddingLayer = tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'linear' });
            this.codeEmbeddingLayer = tf.layers.dense({ units: startDim, inputShape: [codeDim], activation: 'linear' });

            let latentEmb = this.latentEmbeddingLayer.apply(this.gLatent);
            let codeEmb = this.codeEmbeddingLayer.apply(this.gCode);

            // Normalize embeddings
            // latentEmb = tf.layers.batchNormalization().apply(latentEmb);
            // codeEmb = tf.layers.batchNormalization().apply(codeEmb);

            // Normalize with scalar
            latentEmb = new NormalizeLayer({constant: this.latentNorm}).apply(latentEmb);

            this.gEmbedding = tf.layers.add().apply([
                latentEmb,
                codeEmb,
                // this.latentEmbedding.apply(this.gLatent),
                // this.codeEmbedding.apply(this.gCode), 
            ]);
            
            this.backbone = tf.sequential();

            this.backbone.add(tf.layers.dense({ units: startDim, inputShape: [startDim], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                this.backbone.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
            }
    
            this.backbone.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Output layer for 2D points
            this.gOutput = this.backbone.apply(this.gEmbedding);

            this.generator = tf.model({ inputs: [this.gLatent, this.gCode], outputs: this.gOutput });
        }
    
        // Common base network for Discriminator and Q Network
        buildBaseNetwork(numLayers = 4, startDim = 512) {
            this.baseNetwork = tf.sequential();
            this.baseNetwork.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                this.baseNetwork.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }
            this.baseOutput = this.baseNetwork.apply(this.gOutput);
        }
    
        buildQNetwork(codeDim) {
            const inputShape = this.baseOutput.shape.slice(1);
            this.qNetwork = tf.sequential();
            this.qNetwork.add(tf.layers.dense({ units: codeDim, inputShape: inputShape, activation: 'softmax' }));
            this.qNetworkOutput = this.qNetwork.apply(this.baseOutput);
        }
    
        buildDiscriminator(numLayers = 4, startDim = 512, codeDim) {
            // Discriminator output layer
            const inputShape = this.baseOutput.shape.slice(1);
            this.discriminator = tf.sequential();
            this.discriminator.add(tf.layers.dense({ units: 1, inputShape: inputShape, activation: 'sigmoid' }));
            this.discriminatorOutput = this.discriminator.apply(this.baseOutput);
        }
    
        async resetParams() {
            await Promise.all([
                resetWeights(this.generator),
                resetWeights(this.discriminator),
                resetWeights(this.qNetwork)
            ]);
        }
    
        async trainToggle(data, callback = null) {
            if (this.isTraining) {
                this.isTraining = false;
                return;
            }
    
            this.isTraining = true;
            let iter = 0;
    
            const optimizerD = tf.train.adam(0.0005);
            const optimizerG = tf.train.adam(0.0001);
    

            const realLabels = tf.ones([this.batchSize, 1]);
            const fakeLabels = tf.zeros([this.batchSize, 1]);
            const dLabels = tf.concat([realLabels, fakeLabels]);

            while (this.isTraining) { 
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const logValues = tf.tidy(() => {
                    const realSamples = this.realSamplesBuff.toTensor();
                    // const fakeSamples = generateFakeSamples(this.generator, this.latentDim, this.codeDim, this.batchSize);
                    const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
                    const idxs = randInt(0, this.codeDim, this.batchSize);
                    const gCode = tf.oneHot(idxs, this.codeDim);
                    // const gInput = tf.concat([latentPoints, codes], 1);
                    const fakeSamples = this.generator.predict([gLatent, gCode]);

                    const dInputs = tf.concat([realSamples, fakeSamples]);
        
                    // Train D 
                    this.generator.trainable = false;
                    this.baseNetwork.trainable = true;
                    this.qNetwork.trainable = false;
                    this.discriminator.trainable = true;
        
                    const logValues = { iter, gLoss: 0, dLoss: 0, qLoss: 0 };

                    optimizerD.minimize(() => {
                        const baseOutput = this.baseNetwork.predict(dInputs);
                        const dOutput = this.discriminator.predict(baseOutput);
                        const dLossVal = tf.metrics.binaryCrossentropy(dLabels, dOutput).mean();
                        logValues.dLoss = dLossVal.arraySync();

                        const qInput = tf.slice(baseOutput, [0, 0], [this.batchSize, -1]);
                        const qOutput = this.qNetwork.predict(qInput);
                        const qLossVal = tf.metrics.categoricalCrossentropy(gCode, qOutput).mean();
                        logValues.qLoss = qLossVal.arraySync();
                        const totalLoss = dLossVal.add(qLossVal.mul(this.qWeight));
                        return totalLoss;
                    });
        
                    // Train G and Q network jointly
                    this.generator.trainable = true;
                    this.baseNetwork.trainable = true;
                    this.qNetwork.trainable = true;
                    this.discriminator.trainable = false;
                    
                    optimizerG.minimize(() => {
                        const fakeSamples = this.generator.predict([gLatent, gCode]);
                        const baseOutput = this.baseNetwork.predict(fakeSamples);
                        const dOutput = this.discriminator.predict(baseOutput);
                        const gLossVal = tf.metrics.binaryCrossentropy(realLabels, dOutput).mean();
                        logValues.gLoss = gLossVal.arraySync();

                        const qOutput = this.qNetwork.predict(baseOutput);
                        const qLossVal = tf.metrics.categoricalCrossentropy(gCode, qOutput).mean();
                        const totalLoss = gLossVal.add(qLossVal.mul(this.qWeight));
                        return totalLoss;
                    });

                    return logValues;
                });
                if (callback) await callback(logValues);
                iter++;
            }
            tf.dispose([realLabels, fakeLabels, dLabels]);
            this.isTraining = false;
        }
    
        getOneHotCodes() {
            const codeInts = Array.from(
                { length: this.batchSize },
                () => Math.floor(Math.random() * this.codeDim)
            );
            return tf.oneHot(codeInts, this.codeDim);
        }
    
        generate(nSamples) {
            return tf.tidy(() => {
                const gLatent = tf.randomNormal([nSamples, this.latentDim]);
                const codeInts = Array.from(
                    { length: nSamples },
                    () => Math.floor(Math.random() * this.codeDim)
                );
                const gCode = tf.oneHot(codeInts, this.codeDim);
                const pred = this.generator.predict([gLatent, gCode]);
                const ret = pred.arraySync();
                return ret;
            });
        }
    
        dispose() {
            tf.dispose([this.generator, this.discriminator, this.qNetwork]);
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

            this.isInitialized = false;
        }

        async init() {
            await this.gan.init();
            
            this.callback = async ({iter, gLoss, dLoss, qLoss}) => {
                // [gLoss, dLoss, qLoss] = [gLoss, dLoss, qLoss].map(x => parseFloat(x));

                this.gLossVisor.push({ x: iter, y: gLoss });
                this.dLossVisor.push({ x: iter, y: dLoss });
                this.qLossVisor.push({ x: iter, y: qLoss });
                await this.ddm.plot(this);
                this.fpsCounter.update();

                // wait for 1 sec
                // await new Promise(resolve => setTimeout(resolve, 1000));
            }

            this.gridSize = 20;
            const x = tf.linspace(-1, 1, this.gridSize);
            const y = tf.linspace(-1, 1, this.gridSize);
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

        decisionAndGradientMap() {
            return tf.tidy(() => {
                const points = this.decisionMapInputBuff;
                const res = tf.valueAndGrad(
                    point => {
                        const dInput = this.gan.baseNetwork.predict(point);
                        return this.gan.discriminator.predict(dInput);
                    })(points);

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

            this.gLossVisor.clear();
            this.dLossVisor.clear();
            this.qLossVisor.clear();
            this.ddm.plot(this);
        }
    }

    return { ModelHandler };
})();
