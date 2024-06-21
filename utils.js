
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
        tfvis.visor().close();

        this.numUpdates = 0;
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

        this.chart.update();
        this.numUpdates++;
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

class DynamicContourPlot {
    constructor(svg, xlim = null, ylim = null, zlim = null) {
        this.width = svg.attr("width");
        this.height = svg.attr("height");
        this.svg = svg;
        this.mainGroup = svg.append("g");

        this.color = d3.scaleSequential(d3.interpolateViridis)
            .domain(zlim !== null ? zlim : [0, 1]);

        this.xlim = xlim;
        this.ylim = ylim;
        this.zlim = zlim;
    }

    update(data, shape) {
        const { z } = data;

        let contours;
        // if (this.zlim !== null) {
        //     contours = d3.contours()
        //         .size(shape)
        //         .thresholds(
        //             d3.range(
        //                 this.zlim[0],
        //                 this.zlim[1],
        //                 (this.zlim[1] - this.zlim[0]) / 10
        //             )
        //         )(z);
        // } else {
            contours = d3.contours()
                .size(shape)
                (z);
        // }

        // Clear previous contours
        this.mainGroup.selectAll("path").remove();

        // Add contours to the svg
        this.mainGroup.selectAll("path")
            .data(contours)
            .enter()
            .append("path")
            .attr("d", d3.geoPath(d3.geoIdentity().scale(this.width / shape[0])))
            // Handle xlim and ylim
            .attr("transform", `translate(0, ${this.height}) scale(1, -1)`)
            .attr("fill", d => this.color(d.value))
            .attr("stroke", "#69b3a2")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7);
    }

    bringToFront() {
        this.mainGroup.raise();
    }
}

// DynamicScatterPlot class
class DynamicScatterPlot {
    constructor(svg, color = "#69b3a2", xlim = null, ylim = null) {
        this.width = svg.attr("width");
        this.height = svg.attr("height");
        this.svg = svg;
        this.mainGroup = svg.append("g");

        this.xScale = d3.scaleLinear().range([0, this.width]);
        this.yScale = d3.scaleLinear().range([this.height, 0]);

        this.color = color;
        this.xlim = xlim;
        this.ylim = ylim;
    }

    update(data) {
        const { x, y } = data;

        this.mainGroup.selectAll("circle").remove();

        
        // handle xlim and ylim
        if (this.xlim !== null) {
            this.xScale.domain(this.xlim);
        } else {
            this.xScale.domain(d3.extent(x));
        }
        if (this.ylim !== null) {
            this.yScale.domain(this.ylim);
        } else {
            this.yScale.domain(d3.extent(y));
        }

        this.mainGroup.selectAll("circle")
            .data(x.map((d, i) => ({ x: d, y: y[i] })))
            .enter()
            .append("circle")
            .attr("cx", d => this.xScale(d.x))
            .attr("cy", d => this.yScale(d.y))
            .attr("r", 3)
            .attr("fill", this.color)
            .attr("stroke", "#000")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7);
    }

    bringToFront() {
        this.mainGroup.raise();
    }
}

// DynamicDecisionMap class
class DynamicDecisionMap {
    constructor(div, width = 500, height = 500, xlim = null, ylim = null, zlim = null) {
        console.log(div);
        // Find the svg if doesn't exist create one
        this.svg = d3.select(div).select("svg");
        console.log(this.svg, this.svg.empty());
        if (this.svg.empty()) {
            this.svg = d3.select(div).append("svg")
                .attr("width", width)
                .attr("height", height);
            console.log(this.svg, this.svg.empty());
            
        }

        this.contourPlot = new DynamicContourPlot(this.svg, xlim, ylim, zlim);
        this.realDataPlot = new DynamicScatterPlot(this.svg, "black", xlim, ylim);
        this.fakeDataPlot = new DynamicScatterPlot(this.svg, "orange", xlim, ylim);

    }

    update(data) {
        const { realData, fakeData, decisionMap } = data;

        const { x, y, z } = decisionMap;

        const shape = [y.length, x.length];
        
        if (z.length !== shape[0] * shape[1]) {
            console.log(z.length, shape[0], shape[1]);
            throw new Error("Shape of z does not match inferred shape from x and y");
        }

        this.contourPlot.update({ z: z }, shape);
        this.realDataPlot.update(realData);
        this.fakeDataPlot.update(fakeData);
    }

    bringToFront() {
        this.contourPlot.bringToFront();
        this.realDataPlot.bringToFront();
        this.fakeDataPlot.bringToFront();
    }

    plot(gan) {
        // Randomly select 500 points from the real data
        const realData = d3.shuffle(gan.trainData).slice(0, 100);
        const realDataX = realData.map(p => p[0]);
        const realDataY = realData.map(p => p[1]);
        const fakeData = gan.generate(100);
        const fakeDataX = fakeData.map(p => p[0]);
        const fakeDataY = fakeData.map(p => p[1]);
        const decisionData = gan.decisionMap();
    
        const data = {
            realData: { x: realDataX, y: realDataY },
            fakeData: { x: fakeDataX, y: fakeDataY },
            decisionMap: decisionData
        };
    
        this.update(data);
    }
}