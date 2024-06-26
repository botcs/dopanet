const InfoGANDiagram = (function() {
    function constructDescription(div) {
        const data = {
            title: "InfoGAN",
            description: "InfoGAN is an advanced Generative Adversarial Network (GAN) that extends the standard GAN architecture by incorporating a Q network to learn interpretable latent codes. The architecture of InfoGAN consists of three main components:",
            components: [
                "Generator (G): This network generates fake data samples (X fake) from a random noise vector combined with latent codes.",
                "Discriminator (D): This network differentiates between real data (X real) and fake data (X fake) produced by the Generator.",
                "Classifier (Q): This specialized network learns to predict latent codes (C1', C2', C3') from the generated data (X fake), ensuring that the latent variables capture meaningful and interpretable features."
            ],
            flowDescription: "The Generator (G) receives input from latent codes (C1, C2, C3) along with random noise, producing synthetic data samples (X fake). The Discriminator (D) distinguishes between real (X real) and fake (X fake) data, aiding the Generator in improving the quality of generated samples. The Classifier (Q) takes the fake data (X fake) and outputs the predicted latent codes (C1', C2', C3'), helping to enforce the mutual information between the latent codes and the generated data.",
            
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
            "C1'": { cx: 100, cy: 100, width: 50, height: 30, label: "C1'", fill: "#A3C1DA" },
            "C2'": { cx: 200, cy: 100, width: 50, height: 30, label: "C2'", fill: "#B5CDA3" },
            "C3'": { cx: 300, cy: 100, width: 50, height: 30, label: "C3'", fill: "#DAB08C" },
            "Classifier": { cx: 200, cy: 200, width: 130, height: 30, label: "Classifier (Q)", fill: "#B8B8B8" },
            "X_fake": { cx: 300, cy: 300, width: 70, height: 30, label: "X fake" },
            "X_real": { cx: 500, cy: 300, width: 70, height: 30, label: "X real" },
            "Generator": { cx: 300, cy: 400, width: 130, height: 30, label: "Generator (G)", fill: "#B8B8B8" },
            "Discriminator": { cx: 450, cy: 200, width: 160, height: 30, label: "Discriminator (D)", fill: "#B8B8B8" },
            "real": { cx: 500, cy: 100, width: 50, height: 30, label: "real" },
            "fake": { cx: 400, cy: 100, width: 50, height: 30, label: "fake" },
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
            { source: "Classifier", target: "C1'" },
            { source: "Classifier", target: "C2'" },
            { source: "Classifier", target: "C3'" },
            { source: "X_fake", target: "Classifier" },
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