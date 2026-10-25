import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

const container = document.getElementById("mesh-viewer");
const status = document.getElementById("mesh-viewer-status");
const config = {url: document.querySelector("[data-mesh-url]")?.dataset.meshUrl};

function showStatus(message) {
  if (status) {
    status.textContent = message;
    status.hidden = !message;
  }
}

function createRenderer() {
  try {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x1b1f24, 1); // same as the .mesh-viewer background
    return renderer;
  } catch (error) {
    return null;
  }
}

function start() {
  const renderer = createRenderer();
  if (!renderer) {
    showStatus("WebGL is not available in this browser.");
    return;
  }
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  scene.add(camera);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4450, 1.1));
  const headlight = new THREE.DirectionalLight(0xffffff, 1.6);
  headlight.position.set(0.4, 0.8, 1);
  camera.add(headlight); // the light follows the camera, so every side stays lit

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const material = new THREE.MeshStandardMaterial({
    color: 0x5aa9e6,
    roughness: 0.55,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  let home = null; // camera position and target for "Reset view"

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) {
      return;
    }
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function frame(object) {
    const box = new THREE.Box3().setFromObject(object);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
    const distance = (sphere.radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2)) * 1.05;
    const direction = new THREE.Vector3(1, 0.6, 1).normalize();
    camera.near = sphere.radius / 100;
    camera.far = sphere.radius * 100;
    camera.position.copy(sphere.center).addScaledVector(direction, distance);
    camera.updateProjectionMatrix();
    controls.target.copy(sphere.center);
    controls.update();
    home = { position: camera.position.clone(), target: sphere.center.clone() };
  }

  new GLTFLoader().load(
    config.url,
    (gltf) => {
      const meshes = [];
      gltf.scene.traverse((node) => {
        if (node.isMesh) {
          node.material = material;
          if (!node.geometry.attributes.normal) {
            node.geometry.computeVertexNormals();
          }
          meshes.push(node);
        }
      });
      if (!meshes.length) {
        showStatus("The file contains no mesh.");
        return;
      }
      scene.add(gltf.scene);
      resize();
      frame(gltf.scene);
      showStatus("");
      container.dataset.ready = "true";
    },
    undefined,
    () => showStatus("Could not load the mesh."),
  );

  document.getElementById("viewer-reset")?.addEventListener("click", () => {
    if (home) {
      camera.position.copy(home.position);
      controls.target.copy(home.target);
      controls.update();
    }
  });

  const wireframeButton = document.getElementById("viewer-wireframe");
  wireframeButton?.addEventListener("click", () => {
    material.wireframe = !material.wireframe;
    wireframeButton.setAttribute("aria-pressed", String(material.wireframe));
    wireframeButton.classList.toggle("active", material.wireframe);
  });

  new ResizeObserver(resize).observe(container);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

if (container && config && config.url) {
  start();
}
