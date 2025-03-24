document.querySelector('.fake-news-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const newsContent = document.getElementById('news-content').value;
    const resultText = document.getElementById('result-text');
    const overlay = document.getElementById('overlay');
    const background = document.getElementById('background');
    const isFake = detectFakeNews(newsContent);
    
    resultText.textContent = isFake ? "⚠️ This News is FAKE!" : "✅ This News is REAL!";
    overlay.style.display = 'flex';
    
    if (isFake) {
        background.style.animation = 'blinkRed 1s infinite alternate';
    } else {
        background.style.animation = 'blinkGreen 1s infinite alternate';
    }
    
    // Auto-close after 10 seconds
    setTimeout(closeResult, 10000);
});

function detectFakeNews(content) {
    const fakeKeywords = ["fake", "hoax", "scam", "false", "lie"];
    return fakeKeywords.some(keyword => content.toLowerCase().includes(keyword));
}

function closeResult() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('background').style.animation = '';
}