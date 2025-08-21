const form = document.getElementById('predict-form');
const result = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const payload = {
    age: Number(fd.get('age')),
    gender: fd.get('gender')
  };
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: new URLSearchParams(payload) // form post
    });
    const data = await res.json();
    if (res.ok) {
      if (data.probability !== null) {
        result.textContent = `Prediction: ${data.prediction} — probability: ${Math.round(data.probability*100)}%`;
      } else {
        result.textContent = `Prediction: ${data.prediction}`;
      }
    } else {
      result.textContent = `Error: ${data.detail || data.error}`;
    }
  } catch (err) {
    result.textContent = 'Network/error: ' + err.message;
  }
});
