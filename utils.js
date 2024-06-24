
class VisLogger {
    // A class for logging data to the Visor in real time
    constructor({
        name = "Log",
        tab = "History",
        xLabel = "Iteration",
        yLabel = "Y",
        drawArea = null,
        height = 300,
        maxSize = 150,
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
        if (performance.now() - this.lastUpdate > 50) {
            this.chart.update();
            this.lastUpdate = performance.now();
        }

        this.numItems++;
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