// gan.js

// Generator Model
function buildGenerator(latentDim, numLayers = 4, startDim = 128) {
    const generator = tf.sequential();
    generator.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'relu' }));

    for (let i = 1; i < numLayers; i++) {
        generator.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
    }

    generator.add(tf.layers.dense({ units: 2, activation: 'tanh' })); // Output layer for 2D points
    generator.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
    });
    return generator;
}

// Discriminator Model
function buildDiscriminator(numLayers = 4, startDim = 512) {
    const discriminator = tf.sequential();
    discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));

    for (let i = 1; i < numLayers; i++) {
        discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
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

class VanillaGAN {
    constructor(
        {
            latentDim = 100,
            genLayers = 4,
            genStartDim = 16,
            discLayers = 4,
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

    async train(data) {
        const dataTensor = tf.tensor2d(data);
        const halfBatch = Math.floor(this.batchSize / 2);

        for (let iter = 0; iter < this.numIter; iter++) {
            const idx = tf.randomUniform([halfBatch], 0, data.length, 'int32');
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
        tf.dispose([dataTensor]);
    }

    generate(nSamples) {return tf.tidy(() => {
        const latentPoints = generateLatentPoints(this.latentDim, nSamples);
        return this.generator.predict(latentPoints).arraySync();
    })}

    dispose() {
        tf.dispose([this.generator, this.discriminator, this.gan]);
    }

    // FRONTEND STUFF
    decisionMap(gridSize = 10) {return tf.tidy(() => {
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

    plotDecisionBoundary() {
        // Contour plot
        const data = this.decisionMap();
        const name = 'Decision Boundary';
        const container = tfvis.visor().surface({ name, tab: 'Vanilla GAN' });
        const plotlyData = [{
            x: data.x,
            y: data.y,
            z: data.z,
            type: 'contour',
            colorscale: 'Viridis',
        }];

        const layout = {
            title: name,
            xaxis: {title: 'X', range: [-1, 1]},
            yaxis: {title: 'Y', range: [-1, 1]},
        };
        Plotly.newPlot(container.drawArea, plotlyData, layout);
    }

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
                color: 'red'
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
                color: 'red'
            },
            showlegend: false
        };

        const gradTensor = tf.tensor(grads);
        const norm = gradTensor.norm('euclidean', [2]).mean();
        const scale = 1 / norm.dataSync() / gridRes;
        tf.dispose([gradTensor, norm]);

        const arrowSize = 0.05; // Size of the arrowheads

        for (let i = 0; i < gridRes; i++) {
            for (let j = 0; j < gridRes; j++) {
                const startX = x[i];
                const startY = y[j];
                const endX = startX + scale * grads[i][j][0];
                const endY = startY + scale * grads[i][j][1];

                quiverData.x.push(startX);
                quiverData.y.push(startY);

                quiverArrows.x.push(startX, endX, null);
                quiverArrows.y.push(startY, endY, null);

                // Calculate the angle of the gradient vector
                const angle = Math.atan2(grads[i][j][1], grads[i][j][0]);

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

        Plotly.newPlot(container.drawArea, [quiverData, quiverArrows, arrowheads], layout);
    }

}

let gan;
async function initVanillaGAN() {
    gan = new VanillaGAN();
    await gan.init();
}

async function testVanillaGAN() {
    x = getNormalizedInputData();
    gan.numIter = 10;
    for (let i = 0; i < 100; i++) {
        await gan.train(x);
    
        gan.plotDecisionBoundary();
        gan.plotGradientMap();
    }
    // gan.dispose();
}