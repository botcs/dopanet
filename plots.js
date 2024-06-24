
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

    async update(data) {
        
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

        const xExtent = d3.extent(data, d => d.x);
        const yExtent = d3.extent(data, d => d.y);

        if (this.xlim === null) {
            this.x = d3.scaleLinear().domain(xExtent).range([0, this.width]).clamp(true);
        }
        if (this.ylim === null) {
            this.y = d3.scaleLinear().domain(yExtent).range([this.height, 0]).clamp(true);
        }
        
        const arrowSelection = this.arrows.selectAll("line").data(data);
        const p1 = arrowSelection.enter().append("line")
            .attr("class", "arrow")
            .merge(arrowSelection)
            .attr("x1", d => this.x(d.x))
            .attr("y1", d => this.y(d.y))
            .attr("x2", d => this.x(d.x + d.u))
            .attr("y2", d => this.y(d.y + d.v))
            .transition().duration(0).end();

        const p2 = arrowSelection.exit().remove().transition().duration(0).end();

        const arrowheadSelection = this.arrows.selectAll("path").data(data);
        const p3 = arrowheadSelection.enter().append("path")
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
            })
            .transition().duration(0).end();

        const p4 = arrowheadSelection.exit().remove().transition().duration(0).end();
        this.arrows.raise();

        await Promise.all([p1, p2, p3, p4]);
    }
}


class DynamicContourPlot {
    constructor(svg, xlim = null, ylim = null, zlim = null, colormap = null) {
        this.width = svg.attr("width");
        this.height = svg.attr("height");
        this.svg = svg;
        this.mainGroup = svg.append("g")
            .attr("class", "contour-group");

        if (colormap === null) {
            this.colormap = d3.scaleSequential(d3.interpolateViridis)
                .domain(zlim !== null ? zlim : [0, 1]);
        } else {
            this.colormap = colormap;
        }

        this.xlim = xlim;
        this.ylim = ylim;
        this.zlim = zlim;

        this.colorbarGroup = svg.append("g")
            .attr("class", "colorbar-group")
            .attr("transform", `translate(${this.width-50}, ${this.height*0.1}) scale(1, 0.8)`)
            .attr("visibility", "hidden");

        this._createColorbar();
    }

    async update(z) {
        const shape = [z.length, z[0].length];
        z = z.flat();

        const contours = d3.contours()
            .size(shape)
            .smooth(true)(z);


        // Filter contours based on zlim
        const zlim = this.zlim || [d3.min(z), d3.max(z)];
        const filteredContours = contours.filter(d => d.value >= zlim[0] && d.value <= zlim[1]);
        
        const height = this.height;
        const width = this.width;
        const scaleX = width / shape[1];
        const scaleY = height / shape[0];
        const transform = d3.geoTransform({
            point: function(x, y) {
                this.stream.point(x * scaleX, height - y * scaleY );
            }
        });

        const path = d3.geoPath(transform);

        // const paths = this.mainGroup.selectAll("path").data(filteredContours);
        const paths = this.mainGroup.selectAll("path").data(contours);

        await paths.enter()
            .append("path")
            .merge(paths)
            .attr("d", path)
            .attr("fill", d => this.colormap(d.value))
            .attr("stroke", "#69b3a2")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7)
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
            .transition().duration(0).end();


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
            .domain(this.zlim !== null ? this.zlim : this.colormap.domain())
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
                .attr("stroke-width", 4);

            this.colorbarGroup.append("line")
                .attr("class", "limit-line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", barScale(dataMax))
                .attr("y2", barScale(dataMax))
                .attr("stroke", "red")
                .attr("stroke-width", 4);
        }
        this.colorbarGroup.attr("visibility", "visible");
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

    async update(data) {
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
        const p1 = circles
            .attr("cx", d => this.xScale(d[0]))
            .attr("cy", d => this.yScale(d[1]))
            .attr("r", 3)
            .attr("fill", this.color)
            .attr("stroke", "#000")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7)
            .transition().duration(0).end();

        // Enter new circles
        const p2 = circles.enter()
            .append("circle")
            .attr("cx", d => this.xScale(d[0]))
            .attr("cy", d => this.yScale(d[1]))
            .attr("r", 3)
            .attr("fill", this.color)
            .attr("stroke", "#000")
            .attr("stroke-width", 1)
            .attr("opacity", 0.7)
            .transition().duration(0).end();

        // Remove circles that are no longer needed
        const p3 = circles.exit().remove().transition().duration(0).end();

        await Promise.all([p1, p2, p3]);
    }

    bringToFront() {
        this.mainGroup.raise();
    }
}

// DynamicDecisionMap class
class DynamicDecisionMap {
    constructor({div, xlim = null, ylim = null, zlim = null, colormap = null}={}) {
        // Find the svg if doesn't exist create one
        this.svg = d3.select(div).select("svg");
        if (this.svg.empty()) {
            this.svg = d3.select(div).append("svg");

            // Infer width and height
            const width = parseInt(this.svg.style("width"));
            const height = parseInt(this.svg.style("height"));

            // Set attributes
            this.svg
                .attr("width", width)
                .attr("height", height);
        }

        this.contourPlot = new DynamicContourPlot(this.svg, xlim, ylim, zlim, colormap);
        this.quiverPlot = new DynamicQuiverPlot({svg:this.svg, normalize:"mean"});
        this.realDataPlot = new DynamicScatterPlot(this.svg, "black", xlim, ylim);
        this.fakeDataPlot = new DynamicScatterPlot(this.svg, "orange", xlim, ylim);

    }

    async update(data) {
        const { realData, fakeData, decisionMap, gradientMap } = data;

        const p1 = this.contourPlot.update(decisionMap);
        const p2 = this.quiverPlot.update(gradientMap);
        this.contourPlot.colorbarGroup.raise();
        
        const p3 = this.realDataPlot.update(realData);
        const p4 = this.fakeDataPlot.update(fakeData);

        await Promise.all([p1, p2, p3, p4]);
    }

    bringToFront() {
        this.contourPlot.bringToFront();
        this.quiverPlot.bringToFront();
        this.realDataPlot.bringToFront();
        this.fakeDataPlot.bringToFront();
    }

    async plot(modelHandler) {
        // Randomly select 500 points from the real data
        const realData = d3.shuffle(modelHandler.inputData).slice(0, 200);

        const fakeData = modelHandler.generate(200);
        // const decisionMap = gan.decisionMap();
        // const gradientMap = gan.gradientMap();
        const {decisionMap, gradientMap} = modelHandler.decisionAndGradientMap();
        const data = { realData, fakeData, decisionMap, gradientMap};

        await this.update(data);
    }
}

class DynamicMultiDecisionMap {
    constructor({div, xlim = null, ylim = null, zlim = null, maxMaps = 3}={}) {
        // Extract colors from the image and create gradients
        this.colormaps = [
            d3.interpolateRgb("#D0E5FA", "#B3CDE3"), // Light blue gradient
            d3.interpolateRgb("#D6EFD6", "#A9CFA9"), // Light green gradient
            d3.interpolateRgb("#FAE0D6", "#E1B09E")  // Light orange gradient
        ];

        this.div = div;
        // Find the svg if doesn't exist create one
        // this.svg = d3.select(div).select("svg");
        // if (this.svg.empty()) {
        //     this.svg = d3.select(div).append("svg");
        // }

        this.xlim = xlim;
        this.ylim = ylim;
        this.zlim = zlim;
        this.maxMaps = maxMaps;

        this.decisionMaps = [];

        // Create maxMaps number of DynamicDecisionMap instances in advance
        for (let i = 0; i < this.maxMaps; i++) {
            const zlimStep = 0.9 / this.maxMaps;
            const zlim = [0, zlimStep];
            const colormap = this.colormaps[i % this.colormaps.length];

            const decisionMapInstance = new DynamicDecisionMap({
                div: this.div,
                xlim: this.xlim,
                ylim: this.ylim,
                zlim: zlim,
                colormap: colormap
            });

            this.decisionMaps.push(decisionMapInstance);
        }
    }

    async update(data) {
        const { decisionMaps, gradientMaps, realData, fakeData } = data;
        const numMaps = decisionMaps.length;

        // Update only the necessary number of maps
        for (let i = 0; i < this.maxMaps; i++) {
            if (i < numMaps) {
                const decisionMap = decisionMaps[i];
                const gradientMap = gradientMaps[i];

                await this.decisionMaps[i].update({
                    decisionMap,
                    gradientMap,
                    realData,
                    fakeData
                });
            } else {
                // If there are fewer decision maps than maxMaps, update with empty data
                await this.decisionMaps[i].update({
                    decisionMap: [],
                    gradientMap: [],
                    realData: [],
                    fakeData: []
                });
            }
        }

        // Bring all maps to front in order
        this.decisionMaps.forEach(map => map.bringToFront());
    }

    async plot(modelHandler) {
        // Randomly select 500 points from the real data
        const realData = d3.shuffle(modelHandler.inputData).slice(0, 200);
        const fakeData = modelHandler.generate(200);

        const decisionMaps = modelHandler.decisionMaps(); // Should return an array of decision maps
        const gradientMaps = modelHandler.gradientMaps(); // Should return an array of gradient maps
        const data = { realData, fakeData, decisionMaps, gradientMaps };

        await this.update(data);
    }
}
