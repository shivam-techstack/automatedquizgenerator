/**
 * script.js — Automated Quiz Generator Frontend
 *
 * Features:
 *  - MCQ: click option → instant green (correct) / red (wrong) feedback
 *  - True/False: click True/False button → green/red border + feedback
 *  - Fill Blanks: type answer → submit → correct/incorrect feedback
 *  - Live per-section score tracker
 *  - Final Score summary with breakdown
 */

const API_URL = 'http://127.0.0.1:5000/generate-quiz';

// ─────────────────────────────────────────────
// SCORE STATE
// ─────────────────────────────────────────────
let quizState = {
  mcq:  { total: 0, correct: 0 },
  tf:   { total: 0, correct: 0 },
  fill: { total: 0, correct: 0 }
};

function resetScores() {
  quizState = {
    mcq:  { total: 0, correct: 0 },
    tf:   { total: 0, correct: 0 },
    fill: { total: 0, correct: 0 }
  };
  updateScoreboard();
  document.getElementById('finalScoreBox').classList.add('hidden');
}

function updateScoreboard() {
  const total   = quizState.mcq.total   + quizState.tf.total   + quizState.fill.total;
  const correct = quizState.mcq.correct + quizState.tf.correct + quizState.fill.correct;

  // Per-tab score badges
  document.getElementById('mcq-score-badge').textContent =
    quizState.mcq.total  ? quizState.mcq.correct  + '/' + quizState.mcq.total  : '';
  document.getElementById('tf-score-badge').textContent =
    quizState.tf.total   ? quizState.tf.correct   + '/' + quizState.tf.total   : '';
  document.getElementById('fill-score-badge').textContent =
    quizState.fill.total ? quizState.fill.correct + '/' + quizState.fill.total : '';

  // Global score bar
  const answered = quizState.mcq.total + quizState.tf.total + quizState.fill.total;
  if (answered > 0) {
    document.getElementById('globalScoreBar').classList.remove('hidden');
    document.getElementById('globalScoreText').textContent = correct + ' / ' + answered + ' correct';
    const pct = Math.round((correct / answered) * 100);
    document.getElementById('globalScoreFill').style.width = pct + '%';
    document.getElementById('globalScorePct').textContent = pct + '%';
    const fill = document.getElementById('globalScoreFill');
    fill.className = 'score-fill ' + (pct >= 70 ? 'score-fill-good' : pct >= 40 ? 'score-fill-mid' : 'score-fill-low');
  } else {
    document.getElementById('globalScoreBar').classList.add('hidden');
  }
}

function showFinalScore() {
  const mcqT  = quizState.mcq.total,   mcqC  = quizState.mcq.correct;
  const tfT   = quizState.tf.total,    tfC   = quizState.tf.correct;
  const fillT = quizState.fill.total,  fillC = quizState.fill.correct;
  const total   = mcqT + tfT + fillT;
  const correct = mcqC + tfC + fillC;
  const pct = total ? Math.round((correct / total) * 100) : 0;

  let grade = '', gradeClass = '';
  if (pct >= 90)      { grade = '🏆 Excellent!';      gradeClass = 'grade-excellent'; }
  else if (pct >= 70) { grade = '👍 Good Job!';        gradeClass = 'grade-good'; }
  else if (pct >= 50) { grade = '📚 Keep Practising'; gradeClass = 'grade-mid'; }
  else                { grade = '💪 Try Again!';       gradeClass = 'grade-low'; }

  const box = document.getElementById('finalScoreBox');
  box.innerHTML =
    '<div class="final-score-inner">' +
      '<h2>📊 Final Score</h2>' +
      '<div class="final-big-score ' + gradeClass + '">' + correct + ' / ' + total + '</div>' +
      '<div class="final-pct">' + pct + '% &nbsp;·&nbsp; ' + grade + '</div>' +
      '<div class="final-breakdown">' +
        (mcqT  ? '<div class="breakdown-row"><span>Multiple Choice</span><span class="' + (mcqC===mcqT?'bd-perfect':'') + '">' + mcqC + ' / ' + mcqT + '</span></div>' : '') +
        (tfT   ? '<div class="breakdown-row"><span>True / False</span><span class="' + (tfC===tfT?'bd-perfect':'') + '">' + tfC + ' / ' + tfT + '</span></div>' : '') +
        (fillT ? '<div class="breakdown-row"><span>Fill in Blank</span><span class="' + (fillC===fillT?'bd-perfect':'') + '">' + fillC + ' / ' + fillT + '</span></div>' : '') +
      '</div>' +
      (total === 0 ? '<p class="no-ans-note">Answer some questions first!</p>' : '') +
      '<button class="btn-close-score" onclick="document.getElementById(\'finalScoreBox\').classList.add(\'hidden\')">Close</button>' +
    '</div>';
  box.classList.remove('hidden');
  box.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ─────────────────────────────────────────────
// SAMPLE TEXT
// ─────────────────────────────────────────────
const SAMPLE_TEXT = `The water cycle, also known as the hydrological cycle, describes the continuous movement of water on, above, and below the surface of the Earth. Water evaporates from the oceans and lakes due to solar energy, rises into the atmosphere, and cools to form clouds through condensation. When clouds accumulate enough water droplets, precipitation occurs in the form of rain or snow. This water then flows across land as surface runoff, collecting in rivers and streams that eventually return to the oceans, completing the cycle.\n\nPhotosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the Sun, into chemical energy stored in glucose. Plants absorb carbon dioxide from the atmosphere through tiny pores called stomata. They also absorb water from the soil through their roots. Using the energy from sunlight, the chlorophyll in plant cells converts carbon dioxide and water into glucose and oxygen. The oxygen is released into the atmosphere as a byproduct. Photosynthesis is fundamental to life on Earth because it produces the oxygen we breathe and forms the base of the food chain.\n\nThe human heart is a muscular organ that pumps blood throughout the body. It is located in the chest cavity, slightly to the left of center. The heart has four chambers: the right atrium, right ventricle, left atrium, and left ventricle. Deoxygenated blood enters the right atrium from the body, moves to the right ventricle, and is pumped to the lungs to receive oxygen. Oxygenated blood returns to the left atrium, flows into the left ventricle, and is pumped to the rest of the body. The average adult heart beats approximately 72 times per minute.`;

document.getElementById('loadSample').addEventListener('click', () => {
  document.getElementById('inputText').value = SAMPLE_TEXT;
});

// ─────────────────────────────────────────────
// FILE UPLOAD
// ─────────────────────────────────────────────
document.getElementById('fileInput').addEventListener('change', function () {
  const file = this.files[0];
  if (!file) return;
  document.getElementById('fileName').textContent = file.name;
  const reader = new FileReader();
  reader.onload = (e) => { document.getElementById('inputText').value = e.target.result; };
  reader.readAsText(file);
});

// ─────────────────────────────────────────────
// GENERATE QUIZ (main)
// ─────────────────────────────────────────────
async function generateQuiz() {
  const text    = document.getElementById('inputText').value.trim();
  const numMcq  = parseInt(document.getElementById('numMcq').value)  || 5;
  const numTf   = parseInt(document.getElementById('numTf').value)   || 5;
  const numFill = parseInt(document.getElementById('numFill').value) || 5;

  if (!text) {
    showStatus('Please enter or upload some academic text first.', 'error');
    return;
  }

  const btn = document.getElementById('generateBtn');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  showStatus('Processing text and generating questions...', 'loading');
  document.getElementById('outputSection').classList.add('hidden');
  resetScores();

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, num_mcq: numMcq, num_tf: numTf, num_fill: numFill })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Server error: ' + response.status);
    }

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    renderMCQ(data.mcq || []);
    renderTrueFalse(data.true_false || []);
    renderFillBlanks(data.fill_blanks || []);

    document.getElementById('mcqCount').textContent  = (data.mcq || []).length;
    document.getElementById('tfCount').textContent   = (data.true_false || []).length;
    document.getElementById('fillCount').textContent = (data.fill_blanks || []).length;

    document.getElementById('outputSection').classList.remove('hidden');
    document.getElementById('outputSection').scrollIntoView({ behavior: 'smooth', block: 'start' });

    const total = (data.mcq?.length || 0) + (data.true_false?.length || 0) + (data.fill_blanks?.length || 0);
    showStatus('Quiz generated! ' + total + ' questions created successfully.', 'success');

  } catch (error) {
    showStatus('Error: ' + error.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Quiz';
  }
}


// ══════════════════════════════════════════════════════
// MCQ SECTION
// ══════════════════════════════════════════════════════

function renderMCQ(questions) {
  const container = document.getElementById('mcqQuestions');
  quizState.mcq.total = questions.length;

  if (!questions.length) {
    container.innerHTML = emptyState('No MCQ questions could be generated', 'Try providing more detailed academic text.');
    return;
  }

  container.innerHTML = questions.map(function(q, i) {
    const opts = q.options.map(function(opt, oi) {
      return '<li class="option-item" id="mcq-opt-' + i + '-' + oi + '" onclick="selectOption(' + i + ',' + oi + ',\'' + escapeAttr(q.answer) + '\',' + q.options.length + ')">' +
        '<span class="option-label" id="mcq-lbl-' + i + '-' + oi + '">' + String.fromCharCode(65 + oi) + '</span>' +
        '<span class="option-text">' + escapeHtml(opt.substring(3)) + '</span>' +
        '</li>';
    }).join('');

    return '<div class="q-card" id="mcq-card-' + i + '">' +
      '<div class="q-header">' +
        '<div class="q-number">Question ' + (i+1) + ' of ' + questions.length + ' &middot; Multiple Choice</div>' +
        '<div class="q-status-icon" id="mcq-status-' + i + '"></div>' +
      '</div>' +
      '<div class="q-text">' + escapeHtml(q.question) + '</div>' +
      '<ul class="options-list" id="mcq-opts-' + i + '">' + opts + '</ul>' +
      '<div class="q-feedback hidden" id="mcq-feedback-' + i + '"></div>' +
      '<span class="answer-badge reveal-btn" id="mcq-reveal-' + i + '" onclick="revealMCQ(' + i + ',\'' + escapeAttr(q.answer) + '\',' + q.options.length + ')">Show Answer</span>' +
      '</div>';
  }).join('');
}

function selectOption(qIndex, optIndex, correctAnswer, totalOpts) {
  const card = document.getElementById('mcq-card-' + qIndex);
  if (card.dataset.answered === 'true') return;
  card.dataset.answered = 'true';

  const correctText = correctAnswer.replace(/^[A-D]\.\s*/, '').trim().toLowerCase();
  let userCorrect = false;

  for (let i = 0; i < totalOpts; i++) {
    const optEl   = document.getElementById('mcq-opt-' + qIndex + '-' + i);
    const labelEl = document.getElementById('mcq-lbl-' + qIndex + '-' + i);
    const optText = optEl.querySelector('.option-text').textContent.trim().toLowerCase();
    optEl.style.pointerEvents = 'none';

    if (optText === correctText) {
      optEl.classList.add('mcq-correct');
      labelEl.textContent = '\u2714';
      if (i === optIndex) userCorrect = true;
    } else if (i === optIndex) {
      optEl.classList.add('mcq-wrong');
      labelEl.textContent = '\u2718';
    }
  }

  if (userCorrect) quizState.mcq.correct++;
  updateScoreboard();

  card.classList.add(userCorrect ? 'card-correct' : 'card-wrong');

  const icon = document.getElementById('mcq-status-' + qIndex);
  icon.textContent = userCorrect ? '✅' : '❌';

  const fb = document.getElementById('mcq-feedback-' + qIndex);
  fb.classList.remove('hidden', 'feedback-correct', 'feedback-wrong');
  fb.classList.add(userCorrect ? 'feedback-correct' : 'feedback-wrong');
  fb.innerHTML = userCorrect
    ? '<strong>Correct!</strong> Great job!'
    : '<strong>Incorrect.</strong> The correct answer is highlighted in green below.';

  const rev = document.getElementById('mcq-reveal-' + qIndex);
  rev.textContent = 'Answer: ' + correctAnswer;
  rev.classList.remove('reveal-btn');
  rev.classList.add('show-fill');
  rev.onclick = null;
}

function revealMCQ(index, correctAnswer, totalOpts) {
  const card = document.getElementById('mcq-card-' + index);
  if (card.dataset.answered === 'true') return;
  card.dataset.answered = 'true';

  const correctText = correctAnswer.replace(/^[A-D]\.\s*/, '').trim().toLowerCase();

  for (let i = 0; i < totalOpts; i++) {
    const optEl   = document.getElementById('mcq-opt-' + index + '-' + i);
    const labelEl = document.getElementById('mcq-lbl-' + index + '-' + i);
    const optText = optEl.querySelector('.option-text').textContent.trim().toLowerCase();
    optEl.style.pointerEvents = 'none';
    if (optText === correctText) {
      optEl.classList.add('mcq-correct');
      labelEl.textContent = '\u2714';
    }
  }

  const rev = document.getElementById('mcq-reveal-' + index);
  rev.textContent = 'Answer: ' + correctAnswer;
  rev.classList.remove('reveal-btn');
  rev.classList.add('show-fill');
  rev.onclick = null;
}


// ══════════════════════════════════════════════════════
// TRUE / FALSE SECTION
// ══════════════════════════════════════════════════════

function renderTrueFalse(questions) {
  const container = document.getElementById('tfQuestions');
  quizState.tf.total = questions.length;

  if (!questions.length) {
    container.innerHTML = emptyState('No True/False questions generated', 'Try providing more factual sentences.');
    return;
  }

  container.innerHTML = questions.map(function(q, i) {
    return '<div class="q-card" id="tf-card-' + i + '">' +
      '<div class="q-header">' +
        '<div class="q-number">Question ' + (i+1) + ' of ' + questions.length + ' &middot; True / False</div>' +
        '<div class="q-status-icon" id="tf-status-' + i + '"></div>' +
      '</div>' +
      '<div class="q-text">' + escapeHtml(q.question) + '</div>' +
      '<div class="tf-options" id="tf-opts-' + i + '">' +
        '<button class="tf-btn tf-true-btn" onclick="selectTF(' + i + ',\'True\',\'' + q.answer + '\')">&#10004; True</button>' +
        '<button class="tf-btn tf-false-btn" onclick="selectTF(' + i + ',\'False\',\'' + q.answer + '\')">&#10008; False</button>' +
      '</div>' +
      '<div class="q-feedback hidden" id="tf-feedback-' + i + '"></div>' +
      '<span class="answer-badge reveal-btn" id="tf-reveal-' + i + '" onclick="revealTF(' + i + ',\'' + q.answer + '\')">Show Answer</span>' +
      '</div>';
  }).join('');
}

function selectTF(index, selected, correct) {
  const card = document.getElementById('tf-card-' + index);
  if (card.dataset.answered === 'true') return;
  card.dataset.answered = 'true';

  const isCorrect = selected === correct;
  const trueBtn   = document.querySelector('#tf-opts-' + index + ' .tf-true-btn');
  const falseBtn  = document.querySelector('#tf-opts-' + index + ' .tf-false-btn');
  const selBtn    = selected === 'True' ? trueBtn : falseBtn;
  const otherBtn  = selected === 'True' ? falseBtn : trueBtn;

  selBtn.classList.add(isCorrect ? 'tf-selected-correct' : 'tf-selected-wrong');
  otherBtn.classList.add('tf-muted');
  trueBtn.disabled = falseBtn.disabled = true;

  if (isCorrect) quizState.tf.correct++;
  updateScoreboard();

  card.classList.add(isCorrect ? 'card-correct' : 'card-wrong');

  const icon = document.getElementById('tf-status-' + index);
  icon.textContent = isCorrect ? '✅' : '❌';

  const fb = document.getElementById('tf-feedback-' + index);
  fb.classList.remove('hidden', 'feedback-correct', 'feedback-wrong');
  fb.classList.add(isCorrect ? 'feedback-correct' : 'feedback-wrong');
  fb.innerHTML = isCorrect
    ? '<strong>Correct!</strong> The answer is <strong>' + correct + '</strong>.'
    : '<strong>Incorrect.</strong> You chose <strong>' + selected + '</strong> — correct answer is <strong>' + correct + '</strong>.';

  const rev = document.getElementById('tf-reveal-' + index);
  rev.textContent = 'Answer: ' + correct;
  rev.classList.remove('reveal-btn');
  rev.classList.add(correct === 'True' ? 'show-true' : 'show-false');
  rev.onclick = null;
}

function revealTF(index, answer) {
  const card = document.getElementById('tf-card-' + index);
  if (card.dataset.answered === 'true') return;
  card.dataset.answered = 'true';

  const trueBtn  = document.querySelector('#tf-opts-' + index + ' .tf-true-btn');
  const falseBtn = document.querySelector('#tf-opts-' + index + ' .tf-false-btn');
  trueBtn.disabled = falseBtn.disabled = true;
  trueBtn.classList.add('tf-muted');
  falseBtn.classList.add('tf-muted');

  const rev = document.getElementById('tf-reveal-' + index);
  rev.textContent = 'Answer: ' + answer;
  rev.classList.remove('reveal-btn');
  rev.classList.add(answer === 'True' ? 'show-true' : 'show-false');
  rev.onclick = null;
}


// ══════════════════════════════════════════════════════
// FILL IN THE BLANKS SECTION
// ══════════════════════════════════════════════════════

function renderFillBlanks(questions) {
  const container = document.getElementById('fillQuestions');
  quizState.fill.total = questions.length;

  if (!questions.length) {
    container.innerHTML = emptyState('No fill-blank questions generated', 'Try longer, more informative text.');
    return;
  }

  container.innerHTML = questions.map(function(q, i) {
    return '<div class="q-card" id="fill-card-' + i + '">' +
      '<div class="q-header">' +
        '<div class="q-number">Question ' + (i+1) + ' of ' + questions.length + ' &middot; Fill in the Blank</div>' +
        '<div class="q-status-icon" id="fill-status-' + i + '"></div>' +
      '</div>' +
      '<div class="q-text">' + highlightBlank(escapeHtml(q.question)) + '</div>' +
      '<p class="hint-text">Hint: ' + escapeHtml(q.hint) + '</p>' +
      '<div class="fill-input-row">' +
        '<input type="text" class="fill-input" id="fill-input-' + i + '" placeholder="Type your answer here..." ' +
          'onkeydown="if(event.key===\'Enter\') checkFill(' + i + ',\'' + escapeAttr(q.answer) + '\')" />' +
        '<button class="fill-submit-btn" id="fill-submit-' + i + '" onclick="checkFill(' + i + ',\'' + escapeAttr(q.answer) + '\')">Submit</button>' +
      '</div>' +
      '<div class="q-feedback hidden" id="fill-feedback-' + i + '"></div>' +
      '<span class="answer-badge reveal-btn" id="fill-reveal-' + i + '" onclick="revealFill(' + i + ',\'' + escapeAttr(q.answer) + '\')">Show Answer</span>' +
      '</div>';
  }).join('');
}

function checkFill(index, correctAnswer) {
  const card = document.getElementById('fill-card-' + index);
  if (card.dataset.answered === 'true') return;

  const input     = document.getElementById('fill-input-' + index);
  const userAns   = input.value.trim().toLowerCase();
  const correctLc = correctAnswer.trim().toLowerCase();

  if (!userAns) {
    input.style.borderColor = '#f97316';
    input.placeholder = 'Please type an answer first!';
    setTimeout(function() {
      input.style.borderColor = '';
      input.placeholder = 'Type your answer here...';
    }, 1500);
    return;
  }

  card.dataset.answered = 'true';

  const isCorrect = userAns === correctLc || (correctLc.includes(userAns) && userAns.length >= 3);

  if (isCorrect) quizState.fill.correct++;
  updateScoreboard();

  input.disabled = true;
  input.classList.add(isCorrect ? 'fill-input-correct' : 'fill-input-wrong');
  document.getElementById('fill-submit-' + index).disabled = true;

  card.classList.add(isCorrect ? 'card-correct' : 'card-wrong');

  const icon = document.getElementById('fill-status-' + index);
  icon.textContent = isCorrect ? '✅' : '❌';

  const fb = document.getElementById('fill-feedback-' + index);
  fb.classList.remove('hidden', 'feedback-correct', 'feedback-wrong');
  fb.classList.add(isCorrect ? 'feedback-correct' : 'feedback-wrong');
  fb.innerHTML = isCorrect
    ? '<strong>Correct!</strong> Well done — the answer is <strong>' + correctAnswer + '</strong>.'
    : '<strong>Incorrect.</strong> You wrote "<strong>' + escapeHtml(input.value.trim()) + '</strong>" — the correct answer is <strong>' + correctAnswer + '</strong>.';

  const rev = document.getElementById('fill-reveal-' + index);
  rev.textContent = 'Answer: ' + correctAnswer;
  rev.classList.remove('reveal-btn');
  rev.classList.add('show-fill');
  rev.onclick = null;
}

function revealFill(index, answer) {
  const card = document.getElementById('fill-card-' + index);
  if (card.dataset.answered === 'true') return;
  card.dataset.answered = 'true';

  const input  = document.getElementById('fill-input-' + index);
  const submit = document.getElementById('fill-submit-' + index);
  input.value     = answer;
  input.disabled  = true;
  input.classList.add('fill-input-correct');
  submit.disabled = true;

  const rev = document.getElementById('fill-reveal-' + index);
  rev.textContent = 'Answer: ' + answer;
  rev.classList.remove('reveal-btn');
  rev.classList.add('show-fill');
  rev.onclick = null;
}

function highlightBlank(text) {
  return text.replace(/_______/g, '<strong class="blank-highlight">_______</strong>');
}


// ══════════════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════════════
function switchTab(tabName, el) {
  document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  var panels = { mcq: 'mcqPanel', tf: 'tfPanel', fill: 'fillPanel' };
  document.getElementById(panels[tabName]).classList.add('active');
  el.classList.add('active');
}

// ══════════════════════════════════════════════════════
// COPY ALL
// ══════════════════════════════════════════════════════
function copyQuiz() {
  var mcqs  = document.getElementById('mcqQuestions').innerText;
  var tfs   = document.getElementById('tfQuestions').innerText;
  var fills = document.getElementById('fillQuestions').innerText;
  var full  = '=== MCQ QUESTIONS ===\n' + mcqs + '\n\n=== TRUE/FALSE ===\n' + tfs + '\n\n=== FILL IN BLANKS ===\n' + fills;
  navigator.clipboard.writeText(full).then(function() {
    var btn = document.querySelector('.btn-copy');
    btn.textContent = 'Copied!';
    setTimeout(function() { btn.textContent = 'Copy All'; }, 2000);
  });
}

// ══════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════
function showStatus(msg, type) {
  var el = document.getElementById('statusMsg');
  el.textContent = msg;
  el.className = 'status-msg ' + type;
  el.classList.remove('hidden');
}

function emptyState(title, sub) {
  return '<div class="empty-state"><strong>' + title + '</strong><p>' + sub + '</p></div>';
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  return String(str).replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
