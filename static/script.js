const canvas     = document.getElementById('canvas')
const ctx        = canvas.getContext('2d');
const predictBtn = document.getElementById('predict');
const clearBtn   = document.getElementById('clear');
const result     = document.getElementById('result');

const bars = document.getElementById('bars');
const bigDigit = document.getElementById('big-digit');

for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
        <span class="bar-label">${i}</span>
        <div class="bar-track"><div class="bar-fill" id="bar-${i}"></div></div>
        <span class="bar-value" id="val-${i}">0%</span>
    `;
    bars.appendChild(row);
}


ctx.lineWidth = 30;
ctx.lineCap = 'round'; // better rounded strokes

let drawing = false;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return ;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
});
canvas.addEventListener('mouseup', (e) => {
    drawing = 0;
});


clearBtn.addEventListener('click', () => { 
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});
predictBtn.addEventListener('click', () => {
    const small = document.createElement('canvas');
    small.width = 28;
    small.height = 28;
    const smallCtx = small.getContext('2d');
    smallCtx.imageSmoothingEnabled = true;
    smallCtx.imageSmoothingQuality = 'high';
    smallCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = smallCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    const pixels = [];
    for (let i = 0; i < data.length; i += 4)
        pixels.push(data[i] / 255);

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels: pixels })
    })
    .then(response => response.json())
    .then(result => {
        const probs = result.probabilities;
        const maxIdx = probs.indexOf(Math.max(...probs));
        bigDigit.textContent = maxIdx;
        
        for (let i = 0; i < 10; i++) {
            const fill = document.getElementById(`bar-${i}`);
            const val = document.getElementById(`val-${i}`);
            fill.style.width = `${probs[i] * 100}%`;
            val.textContent = `${(probs[i] * 100).toFixed(1)}%`;
            fill.classList.toggle('dim', i !== maxIdx);
        }
    });
    
});

