"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function ThreeHero({ mode = "knot" }) {
  const mountRef = useRef(null);
  const modeRef = useRef(mode);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    const mountNode = mountRef.current;
    if (!mountNode) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0d0a06");

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
    camera.position.z = 4;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountNode.appendChild(renderer.domElement);

    const light = new THREE.PointLight("#f3c86d", 1.35, 20);
    light.position.set(2, 3, 4);
    scene.add(light);
    scene.add(new THREE.AmbientLight("#d7be85", 0.36));

    const stars = new THREE.BufferGeometry();
    const starCount = 700;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i += 1) {
      const idx = i * 3;
      starPositions[idx] = (Math.random() - 0.5) * 14;
      starPositions[idx + 1] = (Math.random() - 0.5) * 14;
      starPositions[idx + 2] = (Math.random() - 0.5) * 14;
    }
    stars.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({ color: "#d8c9a4", size: 0.03 });
    const starField = new THREE.Points(stars, starMaterial);
    scene.add(starField);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.65, 0.03, 32, 200),
      new THREE.MeshStandardMaterial({ color: "#84631f", roughness: 0.42, metalness: 0.24 })
    );
    ring.rotation.x = Math.PI / 2.5;
    scene.add(ring);

    const geometry = new THREE.TorusKnotGeometry(0.95, 0.28, 180, 24);
    const material = new THREE.MeshStandardMaterial({
      color: "#c89b39",
      metalness: 0.5,
      roughness: 0.28
    });
    const knot = new THREE.Mesh(geometry, material);
    scene.add(knot);

    const onResize = () => {
      const width = mountNode.clientWidth;
      const height = Math.max(280, mountNode.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    onResize();
    window.addEventListener("resize", onResize);

    let frame;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const animate = () => {
      if (!prefersReducedMotion) {
        if (modeRef.current === "knot") {
          knot.rotation.x += 0.0045;
          knot.rotation.y += 0.0065;
          knot.scale.lerp(new THREE.Vector3(1, 1, 1), 0.08);
          material.color.lerp(new THREE.Color("#c89b39"), 0.08);
        } else if (modeRef.current === "orbit") {
          knot.rotation.x += 0.003;
          knot.rotation.y += 0.012;
          knot.scale.lerp(new THREE.Vector3(0.78, 0.78, 0.78), 0.08);
          material.color.lerp(new THREE.Color("#e3b44f"), 0.08);
        } else {
          knot.rotation.x += 0.009;
          knot.rotation.y += 0.004;
          knot.scale.lerp(new THREE.Vector3(1.16, 1.16, 1.16), 0.08);
          material.color.lerp(new THREE.Color("#f4d186"), 0.08);
        }
        starField.rotation.y += 0.0006;
        ring.rotation.z += 0.002;
      }
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      geometry.dispose();
      material.dispose();
      ring.geometry.dispose();
      ring.material.dispose();
      stars.dispose();
      starMaterial.dispose();
      renderer.dispose();
      mountNode.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        minHeight: "320px",
        borderRadius: "12px",
        overflow: "hidden",
        border: "1px solid rgba(212, 168, 67, 0.26)"
      }}
    />
  );
}
