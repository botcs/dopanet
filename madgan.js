// MADGAN.js

const MADGAN = (function() {
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

    class MADGAN {
        constructor(
            {
                latentDim = 10,
                codeDim = 3,
                genLayers = 0,
                genStartDim = 256,
                discLayers = 1,
                discStartDim = 256,
                batchSize = 16,
            } = {}
        ) {
            this.generators = [];
            this.discriminator = null;
            this.combinedModel = null;
            this.latentDim = latentDim;
            this.codeDim = codeDim;
            this.genLayers = genLayers;
            this.genStartDim = genStartDim;
            this.discLayers = discLayers;
            this.discStartDim = discStartDim;
            this.batchSize = batchSize;
        }
    
        async init() {
            this.buildGenerators({
                latentDim: this.latentDim, 
                codeDim: this.codeDim, 
                numLayers: this.genLayers, 
                startDim: this.genStartDim
            });
            this.buildDiscriminator({
                numLayers: this.discLayers, 
                startDim: this.discStartDim,
                codeDim: this.codeDim
            });

            this.buildCombinedModels();
            
            this.realSamplesBuff = tf.buffer([this.batchSize, 2]);
            this.codeSamplesBuff = tf.buffer([this.batchSize * this.codeDim]);
            this.fakeSamplesBuff = tf.buffer([this.batchSize * this.codeDim, 2]);
            this.isTraining = false;
        }
    
        buildGenerators({latentDim, numLayers = 4, startDim = 128}) {
            for (let i = 0; i < this.codeDim; i++) {
                const gInput = tf.input({ shape: [latentDim] });
                const backbone = tf.sequential();
                backbone.add(tf.layers.dense({ units: startDim, inputShape: [latentDim], activation: 'relu' }));
        
                for (let i = 1; i < numLayers; i++) {
                    backbone.add(tf.layers.dense({ units: startDim * Math.pow(2, i), activation: 'relu' }));
                }
        
                backbone.add(tf.layers.dense({ units: 2, activation: 'linear' }));
                const gOutput = backbone.apply(gInput);

                const generator = tf.model({ inputs: gInput, outputs: gOutput });
                this.generators.push(generator);
            }
        }
    
        buildDiscriminator({numLayers, startDim, codeDim}={}) {
            const dInput = tf.input({ shape: [2] });
            const discriminator = tf.sequential();
            discriminator.add(tf.layers.dense({ units: startDim, inputShape: [2], activation: 'relu' }));
    
            for (let i = 1; i < numLayers; i++) {
                discriminator.add(tf.layers.dense({ units: startDim / Math.pow(2, i), activation: 'relu' }));
            }

            discriminator.add(tf.layers.dense({ units: codeDim + 1, activation: 'softmax' }));
            const dOutput = discriminator.apply(dInput);

            this.discriminator = tf.model({ inputs: dInput, outputs: dOutput });
            this.discriminator.compile({ 
                optimizer: tf.train.adam(0.001),
                loss: 'sparseCategoricalCrossentropy' 
            });
        }

        buildCombinedModels() {
            this.combinedModels = [];
            for (let i = 0; i < this.codeDim; i++) {
                const gInput = this.generators[i].input;
                const gOutput = this.generators[i].output;
                const dOutput = this.discriminator.apply(gOutput);

                const combinedModel = tf.model({ inputs: gInput, outputs: dOutput });

                combinedModel.compile({ 
                    optimizer: tf.train.adam(0.0001),
                    loss: 'sparseCategoricalCrossentropy'
                });
                this.combinedModels.push(combinedModel);
            }
        }


        readTrainingBuffer() {
            const realData = this.realSamplesBuff.toTensor();
            const fakeData = this.fakeSamplesBuff.toTensor();
            const codeData = this.codeSamplesBuff.toTensor();

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
            const realLabels = tf.fill([this.batchSize, 1], this.codeDim);
            const fakeLabelsJS = [];
            for (let i = 0; i < this.codeDim; i++) {
                fakeLabelsJS.push(tf.fill([this.batchSize, 1], i));
            }
            const fakeLabels = tf.concat(fakeLabelsJS);
            const fakeLabelsVal = fakeLabels.arraySync();
            // save to codeSamplesBuff
            for (let i = 0; i < this.batchSize * this.codeDim; i++) {
                this.codeSamplesBuff.set(fakeLabelsVal[i], i);
            }

            const dLabels = tf.concat([realLabels, fakeLabels]);
    
            while (this.isTraining) { 
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.set(x, i, 0);
                    this.realSamplesBuff.set(y, i, 1);
                }
                const realSamples = this.realSamplesBuff.toTensor();
                const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
    
                const fakeSamples = tf.concat(this.generators.map(gen => gen.predict(gLatent)));
                const fakeSamplesVal = fakeSamples.arraySync();
    
                for (let i = 0; i < this.batchSize * this.codeDim; i++) {
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][0], i, 0);
                    this.fakeSamplesBuff.set(fakeSamplesVal[i][1], i, 1);
                }
    
                const dInputs = tf.concat([realSamples, fakeSamples]);
    
                const logValues = { iter };
    
                this.discriminator.trainable = true;
                this.generators.forEach(gen => gen.trainable = false);
                const dLoss = await this.discriminator.trainOnBatch(dInputs, dLabels);
                logValues.dLoss = dLoss;
    
                this.discriminator.trainable = false;
                this.generators.forEach(gen => gen.trainable = true);
                const gLosses = await Promise.all(
                    this.combinedModels.map((model, i) => {
                        return model.trainOnBatch(gLatent, realLabels);
                    }
                ));
                logValues.gLosses = gLosses;
                logValues.gLossAvg = gLosses.reduce((a, b) => a + b, 0) / this.codeDim;
    
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
            tf.dispose([this.generators, this.discriminator, this.combinedModel]);
        }
    }
    
    class ModelHandler {
        constructor(inputData) {
            this.inputData = inputData;
            this.gan = new MADGAN();
            this.isInitialized = false;
            
            // Build the DOM entry
            this.modelEntry = d3.select('#modelEntryContainer').append('div')
                .attr('class', 'modelEntry')
                .attr('id', 'MADGAN');

            this.modelEntry
            const modelCard = this.modelEntry.append('div')
                .classed('wrapContainer', true);

            const description = modelCard.append('div')
                .classed('wrappedItem', true);
            
            MADGANDiagram.constructDescription(description);
                
            const diagram = modelCard.append('div')
                .attr('id', 'MADGANDiagram')
                .classed('wrappedItem', true);

            MADGANDiagram.constructDiagram(diagram); 
                
            this.trainToggleButton = description.append('button')
                .text('Train')
                .on('click', () => this.trainToggle());

            description.append('button')
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

            this.discriminatorPlot = new DynamicMultiDecisionMap({
                group: this.discriminatorPlot,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                numMaps: 4,
                gridShape: this.gridshape,
            });

            await this.gan.init();

            this.fpsCounter = new FPSCounter("MADGAN FPS");
            
            this.dLossVisor = new VisLogger({
                name: 'Discriminator Loss',
                tab: 'MADGAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });


            this.gLossAvgVisor = new VisLogger({
                name: 'Average Generator Loss',
                tab: 'MADGAN',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });

            this.gLossVisors = [];
            for (let i = 0; i < this.gan.codeDim; i++) {
                this.gLossVisors.push(new VisLogger({
                    name: `Generator ${i} Loss`,
                    tab: 'MADGAN',
                    xLabel: 'Iteration',
                    yLabel: 'Loss',
                }));
            }

            

            this.callback = async ({iter, dLoss, gLosses, gLossAvg}) => {
                this.dLossVisor.push({x: iter, y: dLoss});
                for (let i = 0; i < this.gan.codeDim; i++) {
                    this.gLossVisors[i].push({x: iter, y: gLosses[i]});
                }
                this.gLossAvgVisor.push({x: iter, y: gLossAvg});
                
                const {realData, codeData, fakeData} = this.gan.readTrainingBuffer();
                const {decisionMaps, gradientMap} = this.DiscriminatorDAGMaps();
                this.discriminatorPlot.update({
                    realData,
                    codeData,
                    fakeData,
                    decisionMaps,
                    gradientMap,
                })


                this.fpsCounter.update();
            }

            this.isInitialized = true;
        }

        generate(nSamples) {
            return this.gan.generate(nSamples);
        }

        // DAG = Decision and Gradient
        DiscriminatorDAGMaps() {
            const points = this.decisionMapInputBuff;
            const preds = this.gan.discriminator.predict(points);
            const predsT = preds.transpose();
            
            const grad = tf.grad((point) => {
                const predictions = this.gan.discriminator.predict(point);
                // Gather the last element of each row
                const lastValues = predictions.gather(predictions.shape[1] - 1, 1);
                return lastValues.sum();
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
            // change the button text
            if (this.gan.isTraining) {
                this.trainToggleButton.text('Stop Training');
            } else {
                this.trainToggleButton.text('Resume Training');
            }
        }

        async reset() {
            this.gan.reset();
            this.gLossVisors.forEach(visor => visor.clear());
            this.gLossAvgVisor.clear();
            this.dLossVisor.clear();
        }
    }

    return { ModelHandler };
})();
