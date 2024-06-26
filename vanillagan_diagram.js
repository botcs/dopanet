const VanillaGANDiagram = (function() {
    function constructDescription(div) {
        const descriptionData = {
            title: "Vanill GAN",
            description: "The original Generative Adversarial Network (GAN) consists of two neural networks: a Generator and a Discriminator. The Generator generates fake data samples, while the Discriminator distinguishes between real and fake data samples.",
            components: [
                "Generator (G): This network generates fake data samples (X fake) from a random noise vector.",
                "Discriminator (D): This network differentiates between real data (X real) and fake data (X fake) produced by the Generator.",
            ],
            flowDescription: "The Generator (G) receives random noise, producing synthetic data samples (X fake). The Discriminator (D) distinguishes between real (X real) and fake (X fake) data, aiding the Generator in improving the quality of generated samples.",
            
        };

        // Append the InfoGAN content to the container using D3.js
        const container = div;

        container.append("h1").text(descriptionData.title);

        container.append("p").text(descriptionData.description);

        const ol = container.append("ol");
        descriptionData.components.forEach(component => {
            ol.append("li").html(`<strong>${component.split(':')[0]}</strong>: ${component.split(':')[1]}`);
        });

        container.append("h2").text("Information Flow");
        container.append("p").text(descriptionData.flowDescription);
    }

    function constructDiagram(div) {

        // make it responsive
        const svg = div.append("svg")
            .attr("viewBox", "0 0 650 630");

        const elements = {
            "X_fake": { cx: 300, cy: 300, width: 70, height: 30, label: "X fake" },
            "X_real": { cx: 500, cy: 300, width: 70, height: 30, label: "X real" },
            "Generator": { cx: 300, cy: 400, width: 130, height: 30, label: "Generator (G)", fill: "#B8B8B8" },
            "Discriminator": { cx: 450, cy: 200, width: 160, height: 30, label: "Discriminator (D)", fill: "#B8B8B8" },
            "real": { cx: 500, cy: 100, width: 50, height: 30, label: "real" },
            "fake": { cx: 400, cy: 100, width: 50, height: 30, label: "fake" },
            "z_noise": { cx: 300, cy: 500, width: 70, height: 30, label: "latent" },
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
            { source: "Generator", target: "X_fake" },
            { source: "Discriminator", target: "X_fake" },
            { source: "Discriminator", target: "X_real" },
            { source: "real", target: "Discriminator" },
            { source: "fake", target: "Discriminator" },
            { source: "Generator", target: "z_noise" },
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