import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {methods, runs} from './results.mjs';
import {fitLabels, nodeWidth, textWidth, wrapLabel} from '../app/lib/node-labels.ts';

test('every custom-diagram node label is shown in full, in at most two lines that fit its box', () => {
  for (const method of methods().filter(m => m.architecture?.kind === 'custom')) {
    const labels = method.architecture.nodes.slice(0, 8);
    const cols = labels.length <= 4 ? labels.length : Math.ceil(labels.length / 2);
    const room = nodeWidth(cols) - 10;
    const {size, lines} = fitLabels(labels, room);
    lines.forEach((label, i) => {
      assert.equal(label.join(' '), labels[i], `${method.id}: "${labels[i]}" was shortened`);
      assert.ok(label.length <= 2, `${method.id}: "${labels[i]}" needs ${label.length} lines`);
      for (const line of label) assert.ok(textWidth(line, size) <= room, `${method.id}: "${line}" overflows`);
    });
  }
});

test('labels break between words, never before a lone symbol', () => {
  assert.deepEqual(wrapLabel('HC read · write · mix', 14, 90), ['HC read ·', 'write · mix']);
  assert.deepEqual(wrapLabel('23 blocks', 14, 90), ['23 blocks']);
  const {lines} = fitLabels(['Supercalifragilisticexpialidocious'], 60);
  assert.match(lines[0][0], /…$/);
});

test('every published method has its own model-card figure and atlas preview, not the generic placeholders', () => {
  const read = file => readFileSync(new URL(`../app/${file}`, import.meta.url), 'utf8');
  const sources = {'model-card figure': read('architecture.tsx'), 'atlas preview': read('architecture-preview.tsx')};
  const published = new Set(runs().map(r => r.methodId));
  const special = new Set(['circle']);
  for (const method of methods().filter(m => published.has(m.id))) {
    const supplied = method.architecture?.nodes?.length || method.figureUrl || special.has(method.id);
    for (const [where, source] of Object.entries(sources)) {
      assert.ok(supplied || source.includes(`case '${method.id}'`), `${method.id}: the ${where} falls back to the generic placeholder`);
    }
  }
});
