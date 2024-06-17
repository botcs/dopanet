
class VisLogger {
    // A class for logging data to the Visor in real time
    constructor({
        name = "Log",
        tab = "History",
        xLabel = "Iteration",
        yLabel = "Y",
        height = 300,
        maxSize = 150,
    }) {
        tfvis.visor().close();

        this.numUpdates = 0;
        this.X = [];
        this.Y = [];
        this.yLabel = yLabel;
        this.surface = tfvis.visor().surface({ name: name, tab: tab });
        this.axisSettings = { xLabel: xLabel, yLabel: yLabel, height: height };
        this.maxSize = maxSize;
        this.lastUpdateTime = 0;
        this.timeoutId = null;

        // Create a canvas element for Chart.js
        this.canvas = document.createElement('canvas');
        this.surface.drawArea.appendChild(this.canvas);

        this.chart = new Chart(this.canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: yLabel,
                    data: [],
                    // borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    // fill: false
                }]
            },
            options: {
                responsive: true,
                animation: false,

                plugins: {
                    decimation: {
                        enabled: true,
                        algorithm: 'lttb',
                        samples: this.maxSize,
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: xLabel
                        },
                        ticks: {
                            animation: false // Disable animation for x-axis ticks
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: yLabel
                        },
                        ticks: {
                            animation: false // Disable animation for y-axis ticks
                        }
                    }
                }
            }
        });
        this.chart.data.labels = this.X;
        this.chart.data.datasets[0].data = this.Y;
    }

    push(data) {
        var x, y;
        if (typeof data === "number") {
            x = this.numUpdates;
            y = data;
        } else {
            x = data.x;
            y = data.y;
        }
        this.X.push(x);
        this.Y.push(y);

        // if (this.X.length > this.maxSize) {
        //     const points = this.X.map((x, index) => ({ x, y: this.Y[index] }));
        //     const numReducedPoints = Math.floor(this.maxSize * 0.7);
        //     const reducedPoints = this.largestTriangleThreeBuckets(points, numReducedPoints);

        //     this.X = reducedPoints.map(p => p.x);
        //     this.Y = reducedPoints.map(p => p.y);

        //     this.chart.data.labels = this.X;
        //     this.chart.data.datasets[0].data = this.Y;
        // }

        this.chart.update();
        this.numUpdates++;
    }

    largestTriangleThreeBuckets(data, threshold) {
        if (threshold >= data.length || threshold === 0) {
            return data; // No subsampling needed
        }

        const sampled = [];
        const bucketSize = (data.length - 2) / (threshold - 2);
        let a = 0; // Initially the first point is included
        let maxArea;
        let area;
        let nextA;

        sampled.push(data[a]); // Always include the first point

        for (let i = 0; i < threshold - 2; i++) {
            const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
            const avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
            const avgRange = data.slice(avgRangeStart, avgRangeEnd);
            
            const avgX = avgRange.reduce((sum, point) => sum + point.x, 0) / avgRange.length;
            const avgY = avgRange.reduce((sum, point) => sum + point.y, 0) / avgRange.length;
            
            const rangeStart = Math.floor(i * bucketSize) + 1;
            const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;
            const range = data.slice(rangeStart, rangeEnd);

            maxArea = -1;

            for (let j = 0; j < range.length; j++) {
                area = Math.abs((data[a].x - avgX) * (range[j].y - data[a].y) -
                                (data[a].x - range[j].x) * (avgY - data[a].y));
                if (area > maxArea) {
                    maxArea = area;
                    nextA = rangeStart + j;
                }
            }

            sampled.push(data[nextA]); // Include the point with the largest area
            a = nextA; // Set a to the selected point
        }

        sampled.push(data[data.length - 1]); // Always include the last point
        return sampled;
    }
}

class FPSCounter {
    constructor(name, periodicLog=2000) {
        this.frames = 0;
        this.lastFPS = -1;
        this.startTime = performance.now();

        this.periodicLog = periodicLog;
        this.lastLog = performance.now();

        this.name = name;
        this.vislog = new VisLogger({
            name: name,
            tab: "Debug",
            xLabel: `Time since start (sec)`,
            yLabel: "FPS",
        });
    }

    update() {
        this.frames++;
        const currentTime = performance.now();
        const elapsedTime = currentTime - this.lastLog;
        
        if (currentTime - this.lastLog > this.periodicLog) {
            this.lastFPS = this.frames / (elapsedTime / 1000);
            this.lastLog = currentTime;
            this.log();
            this.frames = 0;
        }
        
    }

    log() {
        // console.log(`${this.name} - moving avg FPS: ${this.lastFPS.toFixed(2)}`);
        // console.log(`Number of tensors: ${tf.memory().numTensors}`);
        let elapsedTime = Math.floor((performance.now() - this.startTime)/1000);
        this.vislog.push({x: elapsedTime, y: this.lastFPS});
    }
}

// PLOTLY
function plotDecisionBoundary(gan) {
    const realData = gan.trainData;
    const realDataX = realData.map(p => p[0]);
    const realDataY = realData.map(p => p[1]);
    const fakeData = gan.generate(500);
    const fakeDataX = fakeData.map(p => p[0]);
    const fakeDataY = fakeData.map(p => p[1]);
    const decisionData = gan.decisionMap();


    // Real data scatter with white color
    const realScatter = {
        x: realDataX,
        y: realDataY,
        mode: "markers",
        type: "scatter",
        name: "Real Data",
        marker: {color: "black", size: 4},
        showlegend: false,
        hovermode: false,
        opacity: 0.8,
    };

    // Fake data scatter with red color
    const fakeScatter = {
        x: fakeDataX,
        y: fakeDataY,
        mode: "markers",
        type: "scatter",
        name: "Fake Data",
        marker: {color: "red", size: 4},
        showlegend: false,
        hovermode: false,
        opacity: 0.8,
    };


    // Contour plot of the decision boundary
    const decisionContour = {
        x: decisionData.x,
        y: decisionData.y,
        z: decisionData.z,
        type: "contour",
        colorscale: "Viridis",
        showscale: false,
        hovermode: false,
        opacity: 0.7,
    };

    const layout = {
        title: "Decision Boundary",
        xaxis: {title: 'X', range: [-1, 1]},
        yaxis: {title: 'Y', range: [-1, 1]},
        width: 800,
        height: 600,
    };
    // use preexisting div
    const plotDiv = document.getElementById("vanillaGAN");
    Plotly.newPlot(
        plotDiv, 
        [decisionContour, realScatter, fakeScatter],
        // [realScatter],
        layout
    );
}

