
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

    update(z) {
        const shape = [z.length, z[0].length];
        // flatten z
        z = z.flat();

        const contours = d3.contours()
                .size(shape)(z);

        // Clear previous contours
        this.mainGroup.selectAll("path").remove();

        // Calculate aspect ratio scaling
        const scale = Math.max(this.width / shape[1], this.height / shape[0]);
        this.mainGroup.selectAll("path")
            .data(contours)
            .enter()
            .append("path")
            .attr("d", d3.geoPath(d3.geoIdentity().scale(scale)))
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
        // handle xlim and ylim
        if (this.xlim !== null) {
            this.xScale.domain(this.xlim);
        } else {
            this.xScale.domain(d3.extent(data, d => d[0]));
        }
        if (this.ylim !== null) {
            this.yScale.domain(this.ylim);
        } else {
            this.yScale.domain(d3.extent(data, d => d[1]));
        }
        // Bind data to existing circles
        const circles = this.mainGroup.selectAll("circle").data(data);

        // console.log(this.xScale.domain(), this.yScale.domain());
        // console.log(this.xScale.range(), this.yScale.range());
        // console.log(this.xScale(data[0][0]), this.yScale(data[0][1]));

        // Update existing circles
        circles
            .attr("cx", d => this.xScale(d[0]))
            .attr("cy", d => this.yScale(d[1]))
            .attr("r", 3)
            .attr("fill", this.color)
            .attr("stroke", "#000")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7);

        // Enter new circles
        circles.enter()
            .append("circle")
            .attr("cx", d => this.xScale(d[0]))
            .attr("cy", d => this.yScale(d[1]))
            .attr("r", 3)
            .attr("fill", this.color)
            .attr("stroke", "#000")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7);

        // Remove circles that are no longer needed
        circles.exit().remove();

    }

    bringToFront() {
        this.mainGroup.raise();
    }
}

// DynamicDecisionMap class
class DynamicDecisionMap {
    constructor(div, xlim = null, ylim = null, zlim = null) {
        // Find the svg if doesn't exist create one
        this.svg = d3.select(div).select("svg");
        if (this.svg.empty()) {
            this.svg = d3.select(div).append("svg");
        }

        // Infer width and height
        const width = parseInt(this.svg.style("width"));
        const height = parseInt(this.svg.style("height"));

        // Set attributes
        this.svg
            .attr("width", width)
            .attr("height", height);


        this.contourPlot = new DynamicContourPlot(this.svg, xlim, ylim, zlim);
        this.realDataPlot = new DynamicScatterPlot(this.svg, "black", xlim, ylim);
        this.fakeDataPlot = new DynamicScatterPlot(this.svg, "orange", xlim, ylim);

    }

    update(data) {
        const { realData, fakeData, decisionMap } = data;

        this.contourPlot.update(decisionMap);
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
        const realData = d3.shuffle(normalizedInputData).slice(0, 200);

        const fakeData = gan.generate(200);
        const decisionData = gan.decisionMap();

        const data = {
            realData: realData,
            fakeData: fakeData,
            decisionMap: decisionData
        };

        this.update(data);
    }
}