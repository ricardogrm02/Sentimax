import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';

function Model(props) {
  const { scene } = useGLTF('/Blu - Animated.glb'); // Correct path for Vite's public folder
  return <primitive object={scene} {...props} />;
}

const ThreeDModel = () => {
  return (
    <Canvas>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />

      {/* Load and render the 3D model */}
      <Model scale={0.5} />

      {/* Controls for rotating/zooming */}
      <OrbitControls />
    </Canvas>
  );
};

export default ThreeDModel;
