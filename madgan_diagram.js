const MADGANDiagram = (function() {
    function constructDescription(div) {
        const data = {
            title: "MAD-GAN",
            description: "MAD-GAN (Multi-Agent Diverse Generative Adversarial Network) is an advanced GAN architecture designed to generate diverse samples by employing multiple generators and a single discriminator. Each generator in MAD-GAN is responsible for generating distinct types of data samples, enhancing the overall diversity of the generated data.",
            components: [
                "Generators (G1, G2, ..., Gk): These networks generate fake data samples (X_fake) from random noise vectors. Each generator is tasked with producing a unique type of data sample, contributing to the diversity of the generated data.",
                "Discriminator (D): This network differentiates between real data (X_real) and fake data (X_fake) produced by the various generators. Additionally, it learns to identify which generator produced each fake sample.",
            ],
            flowDescription: "Each Generator (G1, G2, ..., Gk) receives input from a random noise vector, producing synthetic data samples (X_fake). The Discriminator (D) distinguishes between real (X_real) and fake (X_fake) data and identifies the source generator of each fake sample. This setup aids the generators in improving the quality and diversity of generated samples, ensuring that each generator produces distinct data types."
        };

        // Append the InfoGAN content to the container using D3.js
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
            "fake1": { cx: 100, cy: 100, width: 70, height: 30, label: "fake 1", fill: "#A3C1DA" },
            "fake2": { cx: 200, cy: 100, width: 70, height: 30, label: "fake 2", fill: "#B5CDA3" },
            "fake3": { cx: 300, cy: 100, width: 70, height: 30, label: "fake 3", fill: "#DAB08C" },
            "real": { cx: 450, cy: 100, width: 70, height: 30, label: "real" },
            "xfake1": { cx: 100, cy: 300, width: 70, height: 30, label: "X fake 1", fill: "#A3C1DA" },
            "xfake2": { cx: 200, cy: 300, width: 70, height: 30, label: "X fake 2", fill: "#B5CDA3" },
            "xfake3": { cx: 300, cy: 300, width: 70, height: 30, label: "X fake 3", fill: "#DAB08C" },
            "xreal": { cx: 450, cy: 300, width: 70, height: 30, label: "X real" },
            "g1": { cx: 100, cy: 400, width: 70, height: 30, label: "G1", fill: "#B8B8B8" },
            "g2": { cx: 200, cy: 400, width: 70, height: 30, label: "G2", fill: "#B8B8B8" },
            "g3": { cx: 300, cy: 400, width: 70, height: 30, label: "G3", fill: "#B8B8B8" },

            "Discriminator": { cx: 350, cy: 200, width: 160, height: 30, label: "Discriminator (D)", fill: "#B8B8B8" },
            "z_noise": { cx: 200, cy: 500, width: 70, height: 30, label: "latent" },
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
            { source: "z_noise", target: "g1" },
            { source: "z_noise", target: "g2" },
            { source: "z_noise", target: "g3" },
            { source: "g1", target: "xfake1" },
            { source: "g2", target: "xfake2" },
            { source: "g3", target: "xfake3" },
            { source: "xfake1", target: "Discriminator" },
            { source: "xfake2", target: "Discriminator" },
            { source: "xfake3", target: "Discriminator" },
            { source: "xreal", target: "Discriminator" },
            { source: "Discriminator", target: "real" },
            { source: "Discriminator", target: "fake1" },
            { source: "Discriminator", target: "fake2" },
            { source: "Discriminator", target: "fake3" },
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