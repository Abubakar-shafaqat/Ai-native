import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '../components/HomepageFeatures'; // Assuming this exists

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs">
            Start Reading 🚀
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home | ${siteConfig.title}`}
      description="A comprehensive guide to Physical AI & Humanoid Robotics covering foundational concepts, ROS 2, simulation, humanoid mechanics, and human-robot interaction.">
      <HomepageHeader />
      <main>
        <section className={clsx('container', styles.introSection)}>
          <div className="row">
            <div className="col col--12 text--center">
                        <p className="hero__subtitle">
                          This textbook explores Physical AI & Humanoid Robotics, from foundational concepts to advanced topics like ROS 2, Isaac Sim, and LLMs.
                        </p>            </div>
          </div>
        </section>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}