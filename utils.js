
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

class DynamicQuiverPlot {
    constructor({svg, normalize = "mean", xlim = null, ylim = null}={}) {
        this.width = svg.attr("width");
        this.height = svg.attr("height");
        this.normalize = normalize;
        this.svg = svg;
        this.arrows = this.svg.append("g").attr("class", "arrows");
        this.xlim = xlim;
        this.ylim = ylim;
        if (xlim !== null) {
            this.x = d3.scaleLinear().domain(xlim).range([0, this.width]).clamp(true);
        }
        if (ylim !== null) {
            this.y = d3.scaleLinear().domain(ylim).range([this.height, 0]).clamp(true);
        }
    }

    update(data) {
        
        // data is [gridSizeX, gridSizeY, 4]
        const [gridSizeX, gridSizeY] = [data.length, data[0].length];
        
        // flatten it to [gridSizeX*gridSizeY, 4]
        data = data.flat(1);
        

        // map data array to object
        data = data.map(d => ({ x: d[0], y: d[1], u: d[2], v: d[3] }));

        if (this.normalize === "mean") {
            const norm = d3.mean(data, d => Math.sqrt(d.u ** 2 + d.v ** 2));
            const scaleX = 1 / gridSizeX / norm;
            const scaleY = 1 / gridSizeY / norm;
            data = data.map(d => ({ x: d.x, y: d.y, u: d.u * scaleX, v: d.v * scaleY }));
        } else if (this.normalize === "max") {
            const norm = d3.max(data, d => Math.sqrt(d.u ** 2 + d.v ** 2));
            const scaleX = 1 / gridSizeX / norm;
            const scaleY = 1 / gridSizeY / norm;
            data = data.map(d => ({ x: d.x, y: d.y, u: d.u * scaleX, v: d.v * scaleY }));
        }

        const xExtent = d3.extent(data, d => d.x+d.u);
        const yExtent = d3.extent(data, d => d.y+d.v);

        if (this.xlim === null) {
            this.x = d3.scaleLinear().domain(xExtent).range([0, this.width]).clamp(true);
        }
        if (this.ylim === null) {
            this.y = d3.scaleLinear().domain(yExtent).range([this.height, 0]).clamp(true);
        }
        
        const arrowSelection = this.arrows.selectAll("line").data(data);
        arrowSelection.enter().append("line")
            .attr("class", "arrow")
            .merge(arrowSelection)
            .attr("x1", d => this.x(d.x))
            .attr("y1", d => this.y(d.y))
            .attr("x2", d => this.x(d.x + d.u))
            .attr("y2", d => this.y(d.y + d.v));

        arrowSelection.exit().remove();

        const arrowheadSelection = this.arrows.selectAll("path").data(data);
        arrowheadSelection.enter().append("path")
            .attr("class", "arrowhead")
            .merge(arrowheadSelection)
            .attr("d", d => {
                const angle = Math.atan2(-d.v, d.u); // Negate d.v to account for SVG's inverted y-axis
                const headLength = 10;
                const headWidth = 9;
                const x2 = this.x(d.x + d.u);
                const y2 = this.y(d.y + d.v);
                const points = [
                    { x: x2, y: y2 },
                    { x: x2 - headLength * Math.cos(angle - Math.PI / 6), y: y2 - headLength * Math.sin(angle - Math.PI / 6) },
                    { x: x2 - headWidth * Math.cos(angle + Math.PI), y: y2 - headWidth * Math.sin(angle + Math.PI) },
                    { x: x2 - headLength * Math.cos(angle + Math.PI / 6), y: y2 - headLength * Math.sin(angle + Math.PI / 6) },
                    { x: x2, y: y2 }
                ];
                return d3.line()
                    .x(p => p.x)
                    .y(p => p.y)
                    .curve(d3.curveLinear)(points);
            });

        arrowheadSelection.exit().remove();
        this.arrows.raise();
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

        this.colorbarGroup = svg.append("g")
            .attr("class", "colorbar-group")
            .attr("transform", `translate(${this.width-50}, ${this.height*0.1}) scale(1, 0.8)`);

        this._createColorbar();
    }

    update(z) {
        const shape = [z.length, z[0].length];
        z = z.flat();

        const contours = d3.contours()
            .size(shape)
            .smooth(true)(z);

        const scale = Math.max(this.width / shape[1], this.height / shape[0]);
        const path = d3.geoPath(d3.geoIdentity().scale(scale));

        const paths = this.mainGroup.selectAll("path").data(contours);

        paths.enter()
            .append("path")
            .merge(paths)
            .attr("d", path)
            .attr("transform", `translate(0, ${this.height}) scale(1, -1)`)
            .attr("fill", d => this.color(d.value))
            .attr("stroke", "#69b3a2")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7);

        paths.exit().remove();

        this._updateColorbar(d3.extent(z));
    }

    _createColorbar() {
        const colorbarHeight = this.height;
        const colorScale = d3.scaleSequential(d3.interpolateViridis)
            .domain([400, 0]);
        const rectHeight = colorbarHeight / 400;

        const colorbar = this.colorbarGroup.selectAll(".rect")
            .data(d3.range(400))
            .enter()
            .append("rect")
            .attr("class", "rect")
            .attr("y", d => d * rectHeight)
            .attr("x", 0)
            .attr("height", rectHeight)
            .attr("width", 20)
            .attr("fill", d => colorScale(d));

        const barScale = d3.scaleLinear()
            .range([colorbarHeight, 0]);

        this.colorbarAxis = d3.axisRight(barScale)
            .ticks(6);

        this.colorbarGroup.append("g")
            .attr("class", "colorbar-axis")
            .attr("transform", `translate(20, 0)`)
            .call(this.colorbarAxis);
    }

    _updateColorbar(dataRange) {
        const barScale = d3.scaleLinear()
            .domain(this.zlim !== null ? this.zlim : this.color.domain())
            .range([this.height, 0]);

        this.colorbarAxis.scale(barScale);
        this.colorbarGroup.select(".colorbar-axis").call(this.colorbarAxis);

        const [dataMin, dataMax] = dataRange;

        this.colorbarGroup.selectAll(".limit-line").remove();

        if (this.zlim) {
            this.colorbarGroup.append("line")
                .attr("class", "limit-line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", barScale(dataMin))
                .attr("y2", barScale(dataMin))
                .attr("stroke", "red")
                .attr("stroke-width", 2);

            this.colorbarGroup.append("line")
                .attr("class", "limit-line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", barScale(dataMax))
                .attr("y2", barScale(dataMax))
                .attr("stroke", "red")
                .attr("stroke-width", 2);
        }
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
        this.quiverPlot = new DynamicQuiverPlot({svg:this.svg, normalize:"mean"});
        this.realDataPlot = new DynamicScatterPlot(this.svg, "black", xlim, ylim);
        this.fakeDataPlot = new DynamicScatterPlot(this.svg, "orange", xlim, ylim);

    }

    update(data) {
        const { realData, fakeData, decisionMap, gradientMap } = data;

        this.contourPlot.update(decisionMap);
        this.quiverPlot.update(gradientMap);
        this.realDataPlot.update(realData);
        this.fakeDataPlot.update(fakeData);
    }

    bringToFront() {
        this.contourPlot.bringToFront();
        this.quiverPlot.bringToFront();
        this.realDataPlot.bringToFront();
        this.fakeDataPlot.bringToFront();
    }

    plot(gan) {
        // Randomly select 500 points from the real data
        const realData = d3.shuffle(normalizedInputData).slice(0, 200);

        const fakeData = gan.generate(200);
        // const decisionMap = gan.decisionMap();
        // const gradientMap = gan.gradientMap();
        const [decisionMap, gradientMap] = gan.decisionAndGradientMap();

        const data = { realData, fakeData, decisionMap, gradientMap};

        this.update(data);
    }
}