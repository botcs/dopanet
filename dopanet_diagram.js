const DoPaNetDiagram = (function() {
    function constructDescription(div) {
        const data = {
            title: "Domain Partitioning Network (DoPaNet)",
            description: "DoPaNet is an advanced GAN architecture designed to address the issue of mode collapse by incorporating multiple discriminators and a classifier network. The architecture of DoPaNet consists of three main components:",
            components: [
                "Generator (G): This network generates fake data samples from a random noise vector combined with latent codes.",
                "Discriminators (D1, D2, ..., Dn): These networks are used to evaluate the generated samples, each focusing on a different part of the target distribution to ensure full coverage of the real data distribution.",
                "Classifier (Q): This specialized network decides which discriminator the generator should be trained against for each sample, ensuring that different parts of the data distribution are learned by different discriminators."
            ],
            flowDescription: "The Generator (G) receives input from latent codes and random noise, producing synthetic data samples. The Classifier (Q) determines which Discriminator (Dn) to use for evaluating each sample, thus guiding the Generator to cover the entire target distribution. Each Discriminator (Dn) then evaluates the samples, aiding the Generator in improving the quality and diversity of the generated samples."
        };

        // Append the DoPaNet content to the container using D3.js
        const container = div;

        container.append("h1").text(data.title);

        container.append("p").text(data.description);

        const ol = container.append("ol");
        data.components.forEach(component => {
            ol.append("li").html(`<strong>${component.split(':')[0]}</strong>: ${component.split(':')[1]}`);
        });

        container.append("h2").text("Information Flow");
        container.append("p").text(data.flowDescription);
    }


    function constructDiagram(div) {

        // make it responsive
        const svg = div.append("svg")
            .attr("viewBox", "0 0 650 630");

        const elements = {
            "D1": { cx: 100, cy: 100, width: 50, height: 30, label: "D1", fill: "#A3C1DA" },
            "D2": { cx: 200, cy: 100, width: 50, height: 30, label: "D2", fill: "#B5CDA3" },
            "D3": { cx: 300, cy: 100, width: 50, height: 30, label: "D3", fill: "#DAB08C" },
            "Classifier": { cx: 200, cy: 200, width: 130, height: 30, label: "Classifier (Q)", fill: "#B8B8B8" },
            // "X_real": { cx: 100, cy: 300, width: 70, height: 30, label: "X real" },
            "X_fake": { cx: 300, cy: 300, width: 70, height: 30, label: "X fake" },
            "X_real": { cx: 500, cy: 300, width: 70, height: 30, label: "X real" },
            "Generator": { cx: 300, cy: 400, width: 130, height: 30, label: "Generator (G)", fill: "#B8B8B8" },
            "Discriminator": { cx: 500, cy: 100, width: 160, height: 30, label: "Discriminator (D)", fill: "#B8B8B8" },
            "real": { cx: 600, cy: 30, width: 50, height: 30, label: "real" },
            "fake": { cx: 400, cy: 30, width: 50, height: 30, label: "fake" },
            "c_code": { cx: 200, cy: 500, width: 70, height: 30, label: "code" },
            "z_noise": { cx: 400, cy: 500, width: 70, height: 30, label: "latent" },
            "C1": { cx: 150, cy: 600, width: 30, height: 30, label: "C1", fill: "#A3C1DA" },
            "C2": { cx: 200, cy: 600, width: 30, height: 30, label: "C2", fill: "#B5CDA3" },
            "C3": { cx: 250, cy: 600, width: 30, height: 30, label: "C3", fill: "#DAB08C" },
        };

        Object.keys(elements).forEach(key => {
            const elem = elements[key];
            const x = elem.cx - elem.width / 2;
            const y = elem.cy - elem.height / 2;
            const elemGroup = svg.append("g")
                .attr("class", "elemGroup")
                .attr("transform", `translate(${x}, ${y})`)
                .attr("id", key);

            const rect = elemGroup.append("rect")
                .attr("width", elem.width)
                .attr("height", elem.height)
                .attr("class", "box")
                .style("fill", elem.fill || "white");

            // add text to the box
            elemGroup.append("text")
                .attr("x", elem.width / 2)
                .attr("y", elem.height / 2)
                .attr("dy", "0.35em")
                .attr("class", "text")
                .text(elem.label);
            
        });

        // if a code C1, C2, or C3 is clicked, toggle the .inactive class
        // and toggle the corresponding D1, D2, or D3 box
        svg.selectAll(".elemGroup")
            .on("click", function() {
                const elem = d3.select(this);
                const id = elem.attr("id");
                if (id.startsWith("C") || id.startsWith("D")) {
                    const dId = "D" + id[1];
                    const cId = "C" + id[1];
                    const cBox = d3.select("#" + cId).select("rect");
                    const dBox = d3.select("#" + dId).select("rect");
                    cBox.classed("inactive", !cBox.classed("inactive"));
                    dBox.classed("inactive", !dBox.classed("inactive"));
                }
            });
            
        const links = [
            { source: "Classifier", target: "D1" },
            { source: "Classifier", target: "D2" },
            { source: "Classifier", target: "D3" },
            { source: "X_fake", target: "Classifier" },
            { source: "X_fake", target: "Discriminator" },
            { source: "Generator", target: "X_fake" },
            { source: "Discriminator", target: "X_fake" },
            { source: "Discriminator", target: "X_real" },
            { source: "real", target: "Discriminator" },
            { source: "fake", target: "Discriminator" },
            { source: "Generator", target: "c_code" },
            { source: "Generator", target: "z_noise" },
            { source: "c_code", target: "C1" },
            { source: "c_code", target: "C2" },
            { source: "c_code", target: "C3" },
        ];
            
        const curvePath = (d) => {
            const source = elements[d.source];
            const target = elements[d.target];
            const x1 = source.cx;
            const y1 = source.cy;
            const x2 = target.cx;
            const y2 = target.cy;
            const mx = (x1 + x2) / 2;
            const my = (y1 + y2) / 2;
            return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
        };

        links.forEach(link => {
            svg.append("path")
                .attr("d", curvePath(link))
                .attr("fill", "none")
                .attr("stroke", "black")
                .attr("stroke-width", 2)
                .lower();
        });
    }

    return { constructDescription, constructDiagram };
})();