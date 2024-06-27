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
                latentDim = 10,
                codeDim = 3,
                genLayers = 1,
                genStartDim = 512,
                discLayers = 1,
                discStartDim = 512,
                batchSize = 16,
                qWeight = 0.1,
                gLR = 0.0002,
                dLR = 0.0005,
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
            this.gLR = gLR;
            this.dLR = dLR;

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
            
            // qNetwork.compile({
            //     optimizer: tf.train.adam(this.qLR),
            //     loss: 'categoricalCrossentropy',
            // });
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
                const qLossFn = (yTrue, yPred) => tf.metrics.categoricalCrossentropy(yTrue, yPred).mul(tf.scalar(this.qWeight));

                combinedModel.compile({
                    optimizer: tf.train.adam(this.gLR),
                    loss: [gLossFn, qLossFn],
                });
                this.combinedModels.push(combinedModel);
            }
        }

        async reset() {
            // reset the weights and biases of the generator and discriminator
            
            for (let i = 0; i < this.codeDim; i++) {
                await resetWeights(this.generators[i]);
                await resetWeights(this.discriminators[i]);
            }
            
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
            const qLabelsList = [];
            for (let i = 0; i < this.codeDim; i++) {
                const qLabels = tf.tensor(Array(this.batchSize).fill(i), [this.batchSize], 'int32')
                qLabelsList.push(qLabels);
            }
            // // save to codeSamplesBuff
            // for (let i = 0; i < this.batchSize * this.codeDim; i++) {
            //     this.codeSamplesBuff.set(qLabelsJS[i], i);
            // }

            while (this.isTraining) { 
                this.realSamplesBuff = []
                for (let i = 0; i < this.batchSize; i++) {
                    const [x, y] = data[Math.floor(Math.random() * data.length)];
                    this.realSamplesBuff.push([x, y]);
                }
                const realSamples = tf.tensor(this.realSamplesBuff);
                const gLatent = tf.randomNormal([this.batchSize, this.latentDim]);
                
                // Store values in the buffer for plotting later
                const fakeSamplesList = [];
                this.fakeSamplesBuff = [];
                for (let c = 0; c < this.codeDim; c++) {
                    const fakeSamples = this.generators[c].predictOnBatch(gLatent);
                    fakeSamplesList.push(fakeSamples);
                    const fakeSamplesVal = fakeSamples.arraySync();
                    this.fakeSamplesBuff.push(fakeSamplesVal);
                }

                const logValues = { iter };

                // Train Discriminator
                this.discriminators.forEach(disc => disc.trainable = true);
                this.generators.forEach(gen => gen.trainable = false);
                // evaluate the QNetwork on the real samples
                // and gather them for the corresponding discriminators
                const qRealPred = this.qNetwork.predictOnBatch(realSamples);
                const qRealPredArgmax = tf.argMax(qRealPred, 1);

                // Save QNetwork's predictions on real samples
                const qRealPredArgmaxVal = qRealPredArgmax.arraySync();
                this.qArgmaxBuff = qRealPredArgmaxVal;

                const dLosses = [];
                for (let i = 0; i < this.codeDim; i++) {
                    // gather the real samples for the corresponding discriminator
                    const mask = tf.equal(qRealPredArgmax, i);
                    const dInputReal = await tf.booleanMaskAsync(realSamples, mask);
                    const dInputFake = fakeSamplesList[i];
                    const dInput = tf.concat([dInputReal, dInputFake]);
                    const dLabelReal = tf.fill([dInputReal.shape[0]], 1);
                    const dLabelFake = tf.fill([dInputFake.shape[0]], 0);
                    const dLabels = tf.concat([dLabelReal, dLabelFake]);

                    const discriminator = this.discriminators[i];
                    const dLoss = await discriminator.trainOnBatch(dInput, dLabels);
                    dLosses.push(dLoss);
                    dLosses.push(0);

                    tf.dispose([
                        mask,
                        dInputReal,
                        dInputFake,
                        dInput,
                        dLabelReal,
                        dLabelFake,
                        dLabels
                    ]);
                }
                

                logValues.dLosses = dLosses;
                logValues.avgDLoss = dLosses.reduce((a, b) => a + b) / dLosses.length;


                // Train Generator and Q Network
                this.discriminators.forEach(disc => disc.trainable = false);
                this.generators.forEach(gen => gen.trainable = true);
                const gqLosses = [];
                for (let i = 0; i < this.codeDim; i++) {
                    const combinedModel = this.combinedModels[i];
                    const gLabels = realLabels;
                    const qLabels = qLabelsList[i];
                    const qLabelsOneHot = tf.oneHot(qLabels, this.codeDim);
                    const gqLoss = await combinedModel.trainOnBatch(gLatent, [gLabels, qLabelsOneHot]);
                    gqLosses.push(gqLoss);
                    tf.dispose([qLabelsOneHot]);
                }

                // the gqLosses are
                // 0: joint loss
                // 1: generator-discriminator loss
                // 2: q loss
                logValues.gLosses = gqLosses.map(losses => losses[0]);
                logValues.avgGLoss = logValues.gLosses.reduce((a, b) => a + b) / this.codeDim;
                const qLosses = gqLosses.map(losses => losses[2]);
                logValues.qLoss = qLosses.reduce((a, b) => a + b) / this.codeDim;

                tf.dispose([
                    qRealPred,
                    qRealPredArgmax,
                    realSamples, 
                    fakeSamplesList,
                    gLatent,
                ])
                if (callback) await callback(logValues);
                iter++;
            }
            tf.dispose([realLabels, fakeLabels, ...qLabelsList]);
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
                .attr('id', 'DoPaNet');

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
            this.QNetworkPlotSurface = modelCard.append('div')
                .classed('wrappedItem', true)
                .classed('rowSubplots', true)
                .append('div')
                .classed('plot', true)
                .classed('middlePlot', true)
                .append('svg');
            
            const discriminatorSubplots = modelCard.append('div')
                .classed('wrappedItem', true)
                .classed('rowSubplots', true);
                // .style('grid-column', 'span 2');
                // .style('display', 'grid')
                // .style('grid-template-columns', `repeat(${this.gan.codeDim}, 1fr)`)
                // .style('grid-gap', '10px')


            this.discriminatorPlotSurfaces = [];
            for (let i = 0; i < this.gan.codeDim; i++) {
                const discriminatorPlotSurface = discriminatorSubplots.append('div')
                    .classed('wrappedItem', true)
                    .classed('plot', true)
                    .append('svg');
                this.discriminatorPlotSurfaces.push(discriminatorPlotSurface);
            }


            
        }

        async init() {
            // change the button text
            this.trainToggleButton.text('Initializing...');
            this.gridshape = [10, 10];
            const x = tf.linspace(-1, 1, this.gridshape[0]);
            const y = tf.linspace(-1, 1, this.gridshape[1]);
            const grid = tf.meshgrid(x, y);
            this.decisionMapInputBuff = tf.stack([grid[0].flatten(), grid[1].flatten()], 1);
            tf.dispose([x, y, grid]);

            this.QNetworkPlot = new DynamicMultiDecisionMap({
                group: this.QNetworkPlotSurface,
                xlim: [-1, 1],
                ylim: [-1, 1],
                zlim: [0, 1],
                numMaps: 3,
                gridShape: this.gridshape,
            });


            // Make a discriminator plot for each discriminator
            this.discriminatorPlots = [];
            for (let i = 0; i < this.gan.codeDim; i++) {
                const discriminatorPlot = new DynamicDecisionMap({
                    group: this.discriminatorPlotSurfaces[i],
                    xlim: [-1, 1],
                    ylim: [-1, 1],
                    zlim: [0, 1],
                    gridShape: this.gridshape,
                });
                this.discriminatorPlots.push(discriminatorPlot);
            }


            // this.discriminatorPlot = new DynamicDecisionMap({
            //     group: this.discriminatorPlot,
            //     xlim: [-1, 1],
            //     ylim: [-1, 1],
            //     zlim: [0, 1],
            //     gridShape: this.gridshape,
            // });

            await this.gan.init();

            this.fpsCounter = new FPSCounter("DoPaNet FPS");
            this.dLossAvgVisor = new VisLogger({
                name: 'Average Discriminator Loss',
                tab: 'DoPaNet',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });


            this.gLossAvgVisor = new VisLogger({
                name: 'Average Generator Loss',
                tab: 'DoPaNet',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });

            this.qLossAvgVisor = new VisLogger({
                name: 'Average Q Loss',
                tab: 'DoPaNet',
                xLabel: 'Iteration',
                yLabel: 'Loss',
            });

            this.dLossVisors = [];
            this.gLossVisors = [];
            for (let i = 0; i < this.gan.codeDim; i++) {
                this.dLossVisors.push(new VisLogger({
                    name: `Discriminator ${i} Loss`,
                    tab: 'DoPaNet',
                    xLabel: 'Iteration',
                    yLabel: 'Loss',
                }));
                this.gLossVisors.push(new VisLogger({
                    name: `Generator ${i} Loss`,
                    tab: 'DoPaNet',
                    xLabel: 'Iteration',
                    yLabel: 'Loss',
                }));
            }

            

            this.callback = async (logData) => {
                this.dLossAvgVisor.push({x: logData.iter, y: logData.avgDLoss});
                this.gLossAvgVisor.push({x: logData.iter, y: logData.avgGLoss});
                this.qLossAvgVisor.push({x: logData.iter, y: logData.qLoss});

                for (let i = 0; i < this.gan.codeDim; i++) {
                    this.dLossVisors[i].push({x: logData.iter, y: logData.dLosses[i]});
                    this.gLossVisors[i].push({x: logData.iter, y: logData.gLosses[i]});
                }
                

                const codeData = [];
                for (let c = 0; c < this.gan.codeDim; c++) {
                    codeData.push(Array(this.gan.batchSize).fill(c));
                }


                const {decisionMaps: qdm, gradientMap: qgm} = this.QDAGMaps();
                await this.QNetworkPlot.update({
                    realData: this.gan.realSamplesBuff,
                    codeData: codeData.flat(),
                    fakeData: this.gan.fakeSamplesBuff.flat(),
                    decisionMaps: qdm, 
                    gradientMap: qgm
                });

                for (let c = 0; c < this.gan.codeDim; c++) {
                    const {decisionMap: ddm, gradientMap: dgm} = this.DiscriminatorDAGMaps(c);

                    const selectedRealSamplesBuff = this.gan.realSamplesBuff.filter(
                        (_, idx) => this.gan.qArgmaxBuff[idx] === c
                    );

                    await this.discriminatorPlots[c].update({
                        realData: selectedRealSamplesBuff,
                        codeData: [],
                        fakeData: this.gan.fakeSamplesBuff[c],
                        decisionMap: ddm,
                        gradientMap: dgm
                    });
                }

                this.fpsCounter.update();
            }

            

            this.isInitialized = true;
        }

        generate(nSamples) {
            return this.gan.generate(nSamples);
        }

        // DAG = Decision and Gradient
        DiscriminatorDAGMaps(dIndex) {
            const discriminator = this.gan.discriminators[dIndex];
            const points = this.decisionMapInputBuff;
            const {value: pred, grad} = tf.valueAndGrad(
                point => {
                    return discriminator.predictOnBatch(point);
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
            
            tfvis.visor().setActiveTab('DoPaNet');

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
