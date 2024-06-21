// gan.js

// Generator Model
function buildGenerator(latentDim, numLayers = 4, startDim = 128) {
    const generator = tf.sequential();
    generator.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'linear' }));

    for (let i = 1; i < numLayers; i++) {
        // add batch normalization
        // generator.add(tf.layers.layerNormalization());
        generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'selu'}));
    }

    // generator.add(tf.layers.layerNormalization());
    generator.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Output layer for 2D points
    return generator;
}

// Discriminator Model
function buildDiscriminator(numLayers = 4, startDim = 512) {
    const discriminator = tf.sequential();
    discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'linear' }));

    for (let i = 1; i < numLayers; i++) {
        // discriminator.add(tf.layers.layerNormalization());
        discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'selu'}));
    }

    discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Output layer
    discriminator.compile({
        optimizer: tf.train.adam(0.001),
        // optimizer: tf.train.sgd(0.001),
        loss: 'binaryCrossentropy',
    });
    return discriminator;
}

// Function to generate latent points
function generateLatentPoints(latentDim, nSamples) {
    return tf.randomNormal([nSamples, latentDim]);
}

// Function to generate fake samples
function generateFakeSamples(generator, latentDim, nSamples) {
    const latentPoints = generateLatentPoints(latentDim, nSamples);
    const samples = generator.predict(latentPoints);
    tf.dispose([latentPoints]);
    return samples;
}


async function initVanillaGAN() {
    window.gan = new VanillaGAN();
    await window.gan.init();
}

async function testVanillaGAN() {
    const x = getNormalizedInputData();
    const ddm = new DynamicDecisionMap(
        "#vanillaGANDecisionMap",
        500, 400,
        [-1, 1], [-1, 1],
        [0, 1]
    );
    window.gan.numIter = 1000;
    for (let i = 0; i < 1000; i++) {
        const callback = (iter, gLoss, dLoss) => {
            if (iter % 1 === 0) {
                ddm.plot(gan);
            }
        };
        await window.gan.train(x, callback);
        
    }
}

class VanillaGAN {
    constructor(
        {
            latentDim = 100,
            genLayers = 4,
            genStartDim = 128,
            discLayers = 4,
            discStartDim = 512,
            numIter = 100,
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
        this.numIter = numIter;
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
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
        });

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

        this.fakeSampleBuffer = tf.buffer([this.batchSize, 2]);
        
        this.gridSize = 20;
        const x = tf.linspace(-1, 1, this.gridSize);
        const y = tf.linspace(-1, 1, this.gridSize);
        const grid = tf.meshgrid(x, y);
        this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
        this.decisionMapOutputBuff = tf.buffer([this.gridSize**2, 2]);
        tf.dispose([x, y, grid]);
    }

    async train(trainData, callback = null) {
        this.trainData = trainData;
        const dataTensor = tf.tensor2d(trainData);
        const halfBatch = Math.floor(this.batchSize / 2);

        for (let iter = 0; iter < this.numIter; iter++) {
            const idx = tf.randomUniform([halfBatch], 0, trainData.length, 'int32');
            const realSamples = tf.gather(dataTensor, idx);
            const fakeSamples = generateFakeSamples(this.generator, this.latentDim, halfBatch);

            const realLabels = tf.ones([halfBatch, 1]);
            const fakeLabels = tf.zeros([halfBatch, 1]);

            const dInputs = tf.concat([realSamples, fakeSamples]);
            const dLabels = tf.concat([realLabels, fakeLabels]);

            this.discriminator.trainable = true;
            const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);

            const latentPoints = generateLatentPoints(this.latentDim, this.batchSize);
            const misleadingLabels = tf.ones([this.batchSize, 1]);

            this.discriminator.trainable = false;
            const gLoss = await this.gan.trainOnBatch(latentPoints, misleadingLabels);

            // this.gLossVisor.push({ x: iter, y: gLoss });
            // this.dLossVisor.push({ x: iter, y: dLoss });

            if (callback) callback(iter, gLoss, dLoss);

            tf.dispose([
                idx,
                realSamples, 
                fakeSamples, 
                realLabels, 
                fakeLabels, 
                dInputs,
                dLabels,
                latentPoints, 
                misleadingLabels,
            ]);
        }
    }

    generate(nSamples) {return tf.tidy(() => {
        const latentPoints = generateLatentPoints(this.latentDim, nSamples);
        return this.generator.predict(latentPoints).arraySync();
    })}

    dispose() {
        tf.dispose([this.generator, this.discriminator, this.gan]);
    }

    // FRONTEND STUFF
    decisionMap(gridSize = 20) {
        // const predictions = this.discriminator.predict(points).arraySync();
        
        
        return { x: x.arraySync(), y: y.arraySync(), z: predictions };

    }
        
    gradientMap(gridSize = 20) { return tf.tidy(() => {
        const x = tf.linspace(-1, 1, gridSize);
        const y = tf.linspace(-1, 1, gridSize);
        const grid = tf.meshgrid(x, y);
        const points = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
        
        // Calculate gradients of discriminator output with respect to input points
        const gradients = tf.grad(point => this.discriminator.predict(point))(points);
        const grads = gradients.reshape([gridSize, gridSize, 2]).arraySync();

        return { x: x.arraySync(), y: y.arraySync(), z: grads };
    });}

}
