import os

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Car Racing Game</title>
    <style>
        body {
            margin: 0;
            overflow: hidden;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #111;
            user-select: none;
        }
        #canvas-container {
            width: 100vw;
            height: 100vh;
        }
        #ui {
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            pointer-events: none;
        }
        #speedometer {
            font-size: 36px;
            color: #00ffcc;
        }
        #instructions {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 16px;
            text-align: center;
            pointer-events: none;
        }
        #game-over {
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff3333;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            background: rgba(0,0,0,0.85);
            padding: 30px 50px;
            border-radius: 15px;
            border: 2px solid #ff3333;
            box-shadow: 0 0 20px rgba(255,51,51,0.5);
        }
        #restart-btn {
            margin-top: 20px;
            padding: 10px 25px;
            font-size: 20px;
            background: #ff3333;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: 0.2s;
        }
        #restart-btn:hover {
            background: #ff6666;
            transform: scale(1.05);
        }
    </style>
    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="ui">
        <div>Score: <span id="score">0</span></div>
        <div id="speedometer"><span id="speed">0</span> km/h</div>
    </div>
    <div id="instructions">
        Use <b>LEFT / RIGHT Arrows</b> or <b>A / D</b> to steer. Avoid the other cars!
    </div>
    <div id="game-over">
        GAME OVER
        <div style="font-size: 24px; color: white; margin-top: 10px;">Final Score: <span id="final-score">0</span></div>
        <button id="restart-btn" onclick="resetGame()">PLAY AGAIN</button>
    </div>

    <script>
        let scene, camera, renderer;
        let playerCar;
        let opponentCars = [];
        let roadElements = [];
        let roadSpeed = 0.5;
        let score = 0;
        let gameOver = false;
        let keys = { Left: false, Right: false };
        let playerX = 0;
        const laneWidth = 3;
        const roadLength = 100;

        function init() {
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x222222);
            scene.fog = new THREE.FogExp2(0x222222, 0.015);

            // Camera setup
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 4, 10);
            camera.lookAt(0, 1, -5);

            // Renderer setup
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.getElementById('canvas-container').appendChild(renderer.domElement);

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight.position.set(10, 20, 10);
            dirLight.castShadow = true;
            scene.add(dirLight);

            // Create Road
            createRoad();

            // Create Player Car
            createPlayerCar();

            // Event Listeners
            window.addEventListener('keydown', onKeyDown);
            window.addEventListener('keyup', onKeyUp);
            window.addEventListener('resize', onWindowResize);

            // Start Loop
            animate();
        }

        function createRoad() {
            // Main asphalt road
            const roadGeo = new THREE.PlaneGeometry(12, roadLength);
            const roadMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.8 });
            
            for (let i = 0; i < 3; i++) {
                const road = new THREE.Mesh(roadGeo, roadMat);
                road.rotation.x = -Math.PI / 2;
                road.position.z = -i * roadLength;
                scene.add(road);
                roadElements.push(road);
            }

            // Side barriers / grass representation
            const grassGeo = new THREE.PlaneGeometry(100, roadLength * 3);
            const grassMat = new THREE.MeshStandardMaterial({ color: 0x113311, roughness: 0.9 });
            const grass = new THREE.Mesh(grassGeo, grassMat);
            grass.rotation.x = -Math.PI / 2;
            grass.position.y = -0.05;
            grass.position.z = -roadLength;
            scene.add(grass);
        }

        function createPlayerCar() {
            const carGroup = new THREE.Group();

            // Car Body
            const bodyGeo = new THREE.BoxGeometry(1.6, 0.6, 3);
            const bodyMat = new THREE.MeshStandardMaterial({ color: 0xff0000, metalness: 0.5, roughness: 0.2 });
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 0.4;
            body.castShadow = true;
            carGroup.add(body);

            // Cabin
            const cabinGeo = new THREE.BoxGeometry(1.2, 0.5, 1.5);
            const cabinMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
            const cabin = new THREE.Mesh(cabinGeo, cabinMat);
            cabin.position.set(0, 0.85, -0.2);
            carGroup.add(cabin);

            // Wheels
            const wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.4, 16);
            const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
            
            const wheelPositions = [
                [-0.9, 0.35, 1.0],
                [0.9, 0.35, 1.0],
                [-0.9, 0.35, -1.0],
                [0.9, 0.35, -1.0]
            ];

            wheelPositions.forEach(pos => {
                const wheel = new THREE.Mesh(wheelGeo, wheelMat);
                wheel.rotation.z = Math.PI / 2;
                wheel.position.set(pos[0], pos[1], pos[2]);
                wheel.castShadow = true;
                carGroup.add(wheel);
            });

            playerCar = carGroup;
            scene.add(playerCar);
        }

        function spawnOpponent() {
            if (gameOver) return;

            const carGroup = new THREE.Group();
            const colors = [0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff];
            const randomColor = colors[Math.floor(Math.random() * colors.length)];

            // Car Body
            const bodyGeo = new THREE.BoxGeometry(1.6, 0.6, 3);
            const bodyMat = new THREE.MeshStandardMaterial({ color: randomColor, metalness: 0.4, roughness: 0.3 });
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 0.4;
            carGroup.add(body);

            // Cabin
            const cabinGeo = new THREE.BoxGeometry(1.2, 0.5, 1.5);
            const cabinMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
            const cabin = new THREE.Mesh(cabinGeo, cabinMat);
            cabin.position.set(0, 0.85, -0.2);
            carGroup.add(cabin);

            // Random lane
            const lanes = [-3.5, 0, 3.5];
            const randomLane = lanes[Math.floor(Math.random() * lanes.length)];
            
            carGroup.position.set(randomLane, 0, -150);
            scene.add(carGroup);
            opponentCars.push(carGroup);
        }

        function onKeyDown(e) {
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.Left = true;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.Right = true;
        }

        function onKeyUp(e) {
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.Left = false;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.Right = false;
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        let spawnTimer = 0;

        function animate() {
            if (gameOver) return;

            requestAnimationFrame(animate);

            // Move Road to simulate forward motion
            roadElements.forEach(road => {
                road.position.z += roadSpeed;
                if (road.position.z > roadLength) {
                    road.position.z -= roadLength * 3;
                }
            });

            // Player steering
            if (keys.Left && playerX > -4.5) playerX -= 0.15;
            if (keys.Right && playerX < 4.5) playerX += 0.15;
            playerCar.position.x = playerX;
            playerCar.rotation.y = (keys.Left ? 0.15 : 0) + (keys.Right ? -0.15 : 0);

            // Spawn Opponents
            spawnTimer += 1;
            if (spawnTimer > 80) {
                spawnOpponent();
                spawnTimer = 0;
            }

            // Move and check Opponents
            for (let i = opponentCars.length - 1; i >= 0; i--) {
                const opp = opponentCars[i];
                opp.position.z += roadSpeed + 0.2; // Moving slightly faster/slower relative to road

                // Collision detection (AABB simple check)
                const distZ = Math.abs(opp.position.z - playerCar.position.z);
                const distX = Math.abs(opp.position.x - playerCar.position.x);
                if (distZ < 2.8 && distX < 1.5) {
                    endGame();
                }

                // Remove off-screen cars
                if (opp.position.z > 15) {
                    scene.remove(opp);
                    opponentCars.splice(i, 1);
                    score += 10;
                    document.getElementById('score').innerText = score;
                    // Increase speed gradually
                    roadSpeed += 0.01;
                }
            }

            // Update Speedometer UI
            document.getElementById('speed').innerText = Math.round(roadSpeed * 240);

            renderer.render(scene, camera);
        }

        function endGame() {
            gameOver = true;
            document.getElementById('game-over').style.display = 'block';
            document.getElementById('final-score').innerText = score;
        }

        function resetGame() {
            // Clear opponents
            opponentCars.forEach(opp => scene.remove(opp));
            opponentCars = [];

            // Reset variables
            score = 0;
            roadSpeed = 0.5;
            playerX = 0;
            playerCar.position.x = 0;
            gameOver = false;
            spawnTimer = 0;

            document.getElementById('score').innerText = '0';
            document.getElementById('game-over').style.display = 'none';

            animate();
        }

        // Run Game
        init();
    </script>
</body>
</html>
"""

downloads_path = os.path.expanduser("~/Downloads")
game_file_path = os.path.join(downloads_path, "3D_Car_Racing_Game.html")

with open(game_file_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Game successfully created at: {game_file_path}")
