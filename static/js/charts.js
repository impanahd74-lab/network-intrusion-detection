document.addEventListener('DOMContentLoaded', () => {
  renderMetricsChart();
  renderPredictionSummary();
});

async function renderMetricsChart() {
  const canvas = document.getElementById('metricsChart');
  if (!canvas) {
    return;
  }

  try {
    const response = await fetch('/metrics.json');
    if (!response.ok) {
      displayPlaceholder(canvas, 'Metrics are unavailable. Train models to see results.');
      return;
    }

    const data = await response.json();
    const labels = Object.keys(data.models || {});
    const accuracy = labels.map((label) => data.models[label].accuracy || 0);

    if (labels.length === 0) {
      displayPlaceholder(canvas, 'Metrics are unavailable. Train models to see results.');
      return;
    }

    new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Accuracy',
            data: accuracy,
            backgroundColor: 'rgba(37, 99, 235, 0.6)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
          },
        },
      },
    });
  } catch (error) {
    console.error(error);
    displayPlaceholder(canvas, 'An error occurred while loading metrics.');
  }
}

async function renderPredictionSummary() {
  const canvas = document.getElementById('predictionChart');
  if (!canvas) {
    return;
  }

  try {
    const response = await fetch('/pred_summary.json');
    if (!response.ok) {
      displayPlaceholder(canvas, 'Prediction summary not available. Upload data to predict.');
      return;
    }

    const data = await response.json();
    const labels = Object.keys(data.counts || {});
    const values = labels.map((label) => data.counts[label]);

    if (labels.length === 0) {
      displayPlaceholder(canvas, 'Prediction summary not available. Upload data to predict.');
      return;
    }

    const palette = [
      'rgba(59, 130, 246, 0.7)',
      'rgba(16, 185, 129, 0.7)',
      'rgba(234, 179, 8, 0.7)',
      'rgba(248, 113, 113, 0.7)',
      'rgba(139, 92, 246, 0.7)',
      'rgba(236, 72, 153, 0.7)',
      'rgba(94, 234, 212, 0.7)',
      'rgba(251, 191, 36, 0.7)',
      'rgba(96, 165, 250, 0.7)',
      'rgba(52, 211, 153, 0.7)'
    ];
    const colors = labels.map((_, index) => palette[index % palette.length]);

    new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [
          {
            label: 'Predicted Labels',
            data: values,
            backgroundColor: colors,
            borderWidth: 0,
          },
        ],
      },
    });
  } catch (error) {
    console.error(error);
    displayPlaceholder(canvas, 'An error occurred while loading prediction summary.');
  }
}

function displayPlaceholder(canvas, message) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = '16px "Segoe UI", sans-serif';
  ctx.fillStyle = '#6b7280';
  ctx.textAlign = 'center';
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}
