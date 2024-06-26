// app.js
function main(){
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearCanvasButton = document.getElementById('clearCanvasButton');
    const toggleVisorButton = document.getElementById('toggleVisorButton');
    const toggleTopSectionButton = document.getElementById('toggleTopSectionButton');

    const inputData = [];
    const normalizedInputData = [];

    modelHandlers = {
        'vanilla': new VanillaGAN.ModelHandler(normalizedInputData),
        'infogan': new InfoGAN.ModelHandler(normalizedInputData),
        'madgan': new MADGAN.ModelHandler(normalizedInputData),
    };

    let isDrawing = false;
    let pointsPerSecond = 20;
    let range = 20;
    let lastTime = 0;
    let mouseX = 0;
    let mouseY = 0;

    function generateGaussianPoints(cx, cy, stdDev, numPoints) {
        const points = [];
        for (let i = 0; i < numPoints; i++) {
            const [dx, dy] = truncatedGaussian2D(stdDev);
            points.push([cx + dx, cy + dy]);
        }
        return points;
    }

    function truncatedGaussian2D(stdDev) {
        let u, v, s;
        do {
            u = Math.random() * 2 - 1;
            v = Math.random() * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1 || s === 0);
        const mul = Math.sqrt(-2.0 * Math.log(s) / s);

        // Truncate to 3 standard deviations.
        if (Math.abs(u) > 3) u = 3 * Math.sign(u);
        if (Math.abs(v) > 3) v = 3 * Math.sign(v);
        
        return [stdDev * u * mul, stdDev * v * mul];
    }

    // Drawing
    function scatterPoints(x, y) {
        const newPoints = generateGaussianPoints(x, y, range / 3, pointsPerSecond);
        newPoints.forEach(point => {
            ctx.fillRect(point[0], point[1], 1, 1);
            inputData.push(point);
            normalizedInputData.push([point[0] / canvas.width * 2 - 1, -point[1] / canvas.height * 2 + 1]);
        });
    }

    function drawCircle(x, y, radius) {
        ctx.beginPath();
        ctx.setLineDash([10, 10]);
        // Set the stroke width to 2.
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = 'black';
    }

    function getTouchPos(canvas, touchEvent) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: touchEvent.touches[0].clientX - rect.left,
            y: touchEvent.touches[0].clientY - rect.top
        };
    }

    // Set canvas size and scaling
    const canvasStyle = window.getComputedStyle(canvas);
    canvas.width = parseInt(canvasStyle.width, 10);
    canvas.height = parseInt(canvasStyle.height, 10);


    clearCanvasButton.addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Clear the points array.
        inputData.length = 1;
        normalizedInputData.length = 1;
    });

    toggleVisorButton.addEventListener('click', () => {
        const visor = tfvis.visor();
        visor.toggle();
    });



    canvas.addEventListener('mousedown', (event) => {
        isDrawing = true;
        lastTime = Date.now();
        scatterPoints(event.offsetX, event.offsetY);
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });

    canvas.addEventListener('mousemove', (event) => {
        mouseX = event.offsetX;
        mouseY = event.offsetY;
        if (isDrawing) {
            const currentTime = Date.now();
            const elapsed = currentTime - lastTime;
            if (elapsed > 1000 / pointsPerSecond) {
                scatterPoints(mouseX, mouseY);
                lastTime = currentTime;
            }
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        inputData.forEach(point => ctx.fillRect(point[0], point[1], 1, 1));
        drawCircle(mouseX, mouseY, range);
    });

    canvas.addEventListener('touchstart', (event) => {
        isDrawing = true;
        lastTime = Date.now();
        const touchPos = getTouchPos(canvas, event);
        scatterPoints(touchPos.x, touchPos.y);
    });

    canvas.addEventListener('touchend', () => {
        isDrawing = false;
    });

    canvas.addEventListener('touchmove', (event) => {
        const touchPos = getTouchPos(canvas, event);
        mouseX = touchPos.x;
        mouseY = touchPos.y;
        if (isDrawing) {
            const currentTime = Date.now();
            const elapsed = currentTime - lastTime;
            if (elapsed > 1000 / pointsPerSecond) {
                scatterPoints(mouseX, mouseY);
                lastTime = currentTime;
            }
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        inputData.forEach(point => ctx.fillRect(point[0], point[1], 1, 1));
        drawCircle(mouseX, mouseY, range);
        event.preventDefault();
    });


    function toggleTopSectionHover(){
        // toggle the css of the .bottomSection to overflow-y: unset / auto
        if (document.querySelector('.bottomSection').style['overflow-y'] === 'unset') {
            document.querySelector('.bottomSection').style['overflow-y'] = 'auto';
        } else {
            document.querySelector('.bottomSection').style['overflow-y'] = 'unset';
        }
    }

    toggleTopSectionButton.addEventListener('click', toggleTopSectionHover);

    document.addEventListener('keydown', (event) => {
        if (event.key === 't') {
            toggleTopSectionHover();
        }
    });

    tfvis.visor().close();

    // Set up periodic printing of the number of tensors
    window.numTensorLogger = new VisLogger({
        name: "Number of Tensors over Time",
        tab: "Debug",
        xLabel: "Time",
        yLabel: "Number of Tensors",
    });
    window.startTime = performance.now();
    setInterval(() => {
        const elapsed = Math.floor((performance.now() - window.startTime) / 1000);
        window.numTensorLogger.push({x: elapsed, y: tf.memory().numTensors});
    }, 2000);

    // Set up periodic FPS logging
    window.fps = new FPSCounter("Main Loop FPS", 2000);
    window.fps.start();    
}


document.addEventListener('DOMContentLoaded', main);
