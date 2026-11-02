import { defineConfig, passthroughImageService } from 'astro/config';
import starlight from '@astrojs/starlight';

const REPO = 'https://github.com/Interactive-Active-AI/NextTransformer';
// Deployed under the observatory: https://interactive-active-ai.github.io/NextTransformer/docs/
const base = process.env.DOCS_BASE ?? '/NextTransformer/docs/';
const observatory = process.env.OBSERVATORY_URL ?? 'https://interactive-active-ai.github.io/NextTransformer/';

const group = (label, zh, items, extra = {}) => ({ label, translations: { 'zh-CN': zh }, items, ...extra });

export default defineConfig({
  site: 'https://interactive-active-ai.github.io',
  base,
  image: { service: passthroughImageService() },
  integrations: [
    starlight({
      title: 'NextTransformer',
      description: 'An open framework and fair benchmark for next-generation Transformer architectures.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [
        { icon: 'github', label: 'GitHub', href: REPO },
      ],
      lastUpdated: false,
      customCss: ['./src/styles/custom.css'],
      defaultLocale: 'root',
      locales: {
        root: { label: 'English', lang: 'en' },
        'zh-cn': { label: '简体中文', lang: 'zh-CN' },
      },
      sidebar: [
        { label: 'Observatory ↗', translations: { 'zh-CN': '观测站 ↗' }, link: observatory, attrs: { target: '_self' } },
        group('Start here', '开始', [
          'getting-started/installation', 'getting-started/quick-start', 'getting-started/data',
          'getting-started/models', 'getting-started/repository',
        ]),
        group('Protocol', '协议', [
          'protocol/overview', 'protocol/data-and-evaluation', 'protocol/model-shell', 'protocol/training',
          'protocol/compute', 'protocol/admission', 'protocol/validation', 'protocol/results', 'protocol/versions',
          'protocol/faq',
        ]),
        group('Methods', '方法', ['methods/zoo', 'methods/sources', 'methods/configurations']),
        group('Guides', '指南', [
          'guides/training', 'guides/configuration', 'guides/plugins', 'guides/submit-a-result', 'guides/reproduce',
          'guides/metrics', 'guides/profiling', 'guides/changing-shared-code',
        ]),
        group('Results', '结果', [
          'results/leaderboard', 'results/records', 'results/evidence',
          { slug: 'results/runs', label: 'Run notes', translations: { 'zh-CN': '运行说明' } },
        ]),
        group('Arena', '竞技场', ['arena/overview', 'arena/rules', 'arena/task', 'arena/disclosure', 'arena/circle-loop'],
          { badge: { text: 'v2 · Nov 14', variant: 'note' } }),
        group('Project', '项目', ['roadmap', 'community']),
      ],
    }),
  ],
});
