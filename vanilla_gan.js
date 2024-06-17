// gan.js

// Generator Model
function buildGenerator(latentDim, numLayers = 4, startDim = 128) {
    const generator = tf.sequential();
    generator.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'selu' }));

    for (let i = 1; i < numLayers; i++) {
        // add batch normalization
        // generator.add(tf.layers.layerNormalization());
        generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'selu'}));
    }

    // generator.add(tf.layers.layerNormalization());
    generator.add(tf.layers.dense({ units: 2, activation: 'linear' })); // Output layer for 2D points
    generator.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
    });
    return generator;
}

// Discriminator Model
function buildDiscriminator(numLayers = 4, startDim = 512) {
    const discriminator = tf.sequential();
    discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'selu' }));

    for (let i = 1; i < numLayers; i++) {
        discriminator.add(tf.layers.layerNormalization());
        discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'selu'}));
    }

    discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Output layer
    discriminator.compile({
        optimizer: tf.train.adam(0.001),
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
    x = getNormalizedInputData();
    window.gan.numIter = 1000;
    for (let i = 0; i < 1000; i++) {
        const callback = (iter, gLoss, dLoss) => {
            plotDecisionBoundary(window.gan);
        };
        await window.gan.train(x, callback);
        
    }
}

class VanillaGAN {
    constructor(
        {
            latentDim = 100,
            genLayers = 2,
            genStartDim = 128,
            discLayers = 2,
            discStartDim = 128,
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
            optimizer: tf.train.sgd(0.001),
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

            this.gLossVisor.push({ x: iter, y: gLoss });
            this.dLossVisor.push({ x: iter, y: dLoss });

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
    decisionMap(gridSize = 20) {return tf.tidy(() => {
        const x = tf.linspace(-1, 1, gridSize);
        const y = tf.linspace(-1, 1, gridSize);
        const grid = tf.meshgrid(x, y);
        const points = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
        const predictions = this.discriminator.predict(points).reshape([gridSize, gridSize]).arraySync();
        return { x: x.arraySync(), y: y.arraySync(), z: predictions };
    });}
        
    gradientMap(gridSize = 10) { return tf.tidy(() => {
        const x = tf.linspace(-1, 1, gridSize);
        const y = tf.linspace(-1, 1, gridSize);
        const grid = tf.meshgrid(x, y);
        const points = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
        
        // Calculate gradients of discriminator output with respect to input points
        const gradients = tf.grad(point => this.discriminator.predict(point))(points);
        const grads = gradients.reshape([gridSize, gridSize, 2]).arraySync();

        return { x: x.arraySync(), y: y.arraySync(), z: grads };
    });}

    plotGradientMap() {
        // streamplot
        const data = this.gradientMap();
        const name = 'Gradient Map';
        const container = tfvis.visor().surface({ name, tab: 'Vanilla GAN' });
        
        // Function to plot the gradient field with arrowheads
        const { x, y, z: grads } = data;
        const gridRes = x.length;

        // Prepare the quiver plot data
        const quiverData = {
            type: 'scattergl',
            mode: 'markers',
            x: [],
            y: [],
            marker: { size: .01, color: 'black' },
            showlegend: false
        };

        const quiverArrows = {
            type: 'scattergl',
            mode: 'lines',
            x: [],
            y: [],
            line: {
                width: 3,
                color: 'black'
            },
            showlegend: false
        };

        const arrowheads = {
            type: 'scattergl',
            mode: 'lines',
            x: [],
            y: [],
            line: {
                width: 3,
                color: 'black'
            },
            showlegend: false
        };

        const trainingData = {
            x: getNormalizedInputData().map(p => p[0]),
            y: getNormalizedInputData().map(p => p[1]),
            mode: 'markers',
            marker: { size: 4, color: 'red' },
            showlegend: false
        }

        const avgNorm = tf.tidy(() => {
            const gradTensor = tf.tensor(grads);
            const norm = gradTensor.norm('euclidean', [2]).mean();
            return norm;
        });
        const scale = 1 / avgNorm.dataSync() / gridRes;
        

        const arrowSize = 0.05; // Size of the arrowheads

        for (let i = 0; i < gridRes; i++) {
            for (let j = 0; j < gridRes; j++) {
                const startX = x[i];
                const startY = y[j];
                // Truncate at length 1 after rescaling
                const U = Math.min(grads[i][j][0] * scale, 1);
                const V = Math.min(grads[i][j][1] * scale, 1);
                
                const endX = startX + U;
                const endY = startY + V;

                quiverData.x.push(startX);
                quiverData.y.push(startY);

                quiverArrows.x.push(startX, endX, null);
                quiverArrows.y.push(startY, endY, null);

                // Calculate the angle of the gradient vector
                const angle = Math.atan2(U, V);

                // Calculate the arrowhead points
                const arrowhead1X = endX - arrowSize * Math.cos(angle - Math.PI / 6);
                const arrowhead1Y = endY - arrowSize * Math.sin(angle - Math.PI / 6);
                const arrowhead2X = endX - arrowSize * Math.cos(angle + Math.PI / 6);
                const arrowhead2Y = endY - arrowSize * Math.sin(angle + Math.PI / 6);

                // Add arrowhead lines
                arrowheads.x.push(endX, arrowhead1X, null);
                arrowheads.y.push(endY, arrowhead1Y, null);
                arrowheads.x.push(endX, arrowhead2X, null);
                arrowheads.y.push(endY, arrowhead2Y, null);
            }
        }

        const layout = {
            title: 'Gradient Field',
            xaxis: { title: 'X' },
            yaxis: { title: 'Y', scaleanchor: "x", scaleratio: 1 },
            showlegend: false,
            hovermode: false
        };

        Plotly.react(
            container.drawArea, 
            [
                quiverData, 
                quiverArrows, 
                arrowheads,
                trainingData
            ], 
            layout
        );
    }

}
