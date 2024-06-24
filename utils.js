
class VisLogger {
    // A class for logging data to the Visor in real time
    constructor({
        name = "Log",
        tab = "History",
        xLabel = "Iteration",
        yLabel = "Y",
        drawArea = null,
        height = 300,
        maxSize = 200,
    }) {
        this.numItems = 0;
        this.X = [];
        this.Y = [];
        this.yLabel = yLabel;
        this.axisSettings = { xLabel: xLabel, yLabel: yLabel, height: height };
        this.maxSize = maxSize;
        this.lastUpdateTime = 0;
        this.timeoutId = null;
        
        // Create a canvas element for Chart.js
        this.canvas = document.createElement('canvas');
        if (drawArea) {
            drawArea.appendChild(this.canvas);
        } else {
            this.surface = tfvis.visor().surface({ name: name, tab: tab });
            this.surface.drawArea.appendChild(this.canvas);
        }
        
        this.chart = new Chart(this.canvas, {
            type: 'line',
            data: {
                labels: this.X,
                datasets: [{
                    label: yLabel,
                    data: this.Y,
                    borderWidth: 1,
                }]
            },
            options: {
                responsive: true,
                animation: false,
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
        this.lastUpdate = performance.now();
    }

    push(data) {
        let x, y;
        if (typeof data === "number") {
            x = this.numItems;
            y = data;
        } else {
            x = data.x;
            y = data.y;
        }
        this.X.push(x);
        this.Y.push(y);

        
        if (this.X.length > this.maxSize) {
            const points = this.X.map((x, index) => ({ x, y: this.Y[index] }));
            const numReducedPoints = Math.floor(this.maxSize * 0.7);
            const reducedPoints = this.largestTriangleThreeBuckets(points, numReducedPoints);

            this.X = reducedPoints.map(p => p.x);
            this.Y = reducedPoints.map(p => p.y);

            this.chart.data.labels = this.X;
            this.chart.data.datasets[0].data = this.Y;
        }


        // only update if the surface is open
        if (tfvis.visor().isOpen()) {
            if (performance.now() - this.lastUpdate > 100) {
                this.chart.update();
                this.lastUpdate = performance.now();
            }
        }
        this.numItems++;
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

    clear() {
        this.X.length = 0;
        this.Y.length = 0;
        this.numItems = 0;
        this.chart.update();
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
        let elapsedTime = Math.floor((performance.now() - this.startTime)/1000);
        this.vislog.push({x: elapsedTime, y: this.lastFPS});
    }

    start() {
        // Request animation frame
        const loop = () => {
            this.update();
            requestAnimationFrame(loop);
        }
        loop();
    }
}


function randomNormal(targetTensorBuffer) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    // return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    val = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}


function randInt(min, max, n=1) {
    if (n === 1) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    return Array.from(
        {length: n}, 
        () => Math.floor(Math.random() * (max - min + 1)) + min
    );
}

// Define a custom layer that normalizes the input tensor
class MultiplyLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.constant = config.constant;
    }

    call(inputs) {
        const input = inputs[0];
        return tf.mul(input, tf.scalar(this.constant));
    }

    static get className() {
        return 'MultiplyLayer';
    }
}