import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here

const config: Config = {
  title: 'Hackathon Native Book – Physical AI & Humanoid Robotics',
  tagline: 'A Comprehensive Guide to Building Intelligent Robots',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://panaversity.github.io',
  baseUrl: '/physical-ai-book/',
  organizationName: 'panaversity',
  projectName: 'physical-ai-book',

  onBrokenLinks: 'ignore',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur', 'es'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
        htmlLang: 'en-US',
        calendar: 'gregory',
        path: 'en',
      },
      ur: {
        label: 'اردو',
        direction: 'rtl',
        htmlLang: 'ur-PK',
        calendar: 'gregory',
        path: 'ur',
      },
      es: {
        label: 'Español',
        direction: 'ltr',
        htmlLang: 'es-ES',
        calendar: 'gregory',
        path: 'es',
      },
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/docs',
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',

    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
          to: '/docs',
        },
        {
          type: 'html',
          position: 'right',
          value: '<span style="margin-right:5px;">Translate Book:</span>',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Creator / Developer',
          items: [
            {
              label: 'Noor Ul Sehar',
              href: 'https://github.com/nh7220869',
            },
            {
              label: 'Robotics Engineer | AI Enthusiast',
              to: '#',
            },
          ],
        },
        {
          title: 'External Links',
          items: [
            {
              label: 'Portfolio',
              href: 'https://v1-portfolio-web-next-js-cs--beta.vercel.app/',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/nh7220869',
            },
            {
              label: 'Vercel',
              href: 'https://vercel.com/noor-ul-sehars-projects',
            },
            {
              label: 'Discord',
              href: 'https://discord.com/users/nh7220869',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Developed by Noor Ul Sehar.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
