import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Comprehensive Coverage',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default, // Placeholder Svg
    description: (
      <>
        Explore the foundational concepts and advanced applications of Physical AI and Humanoid Robotics, from theory to practical implementation.
      </>
    ),
  },
  {
    title: 'Practical Insights',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default, // Placeholder Svg
    description: (
      <>
        Gain practical understanding with real-world examples, case studies, and hands-on guidance for building intelligent robotic systems.
      </>
    ),
  },
  {
    title: 'Cutting-Edge Research',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default, // Placeholder Svg
    description: (
      <>
        Stay updated with the latest advancements and research trends in robotics, machine learning, and artificial intelligence for physical systems.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}