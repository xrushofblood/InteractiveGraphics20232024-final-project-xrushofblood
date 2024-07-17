// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	// [TO-DO] Modify the code below to form the transformation matrix.
	//I have inverted the rotation angles to make the object follow the movement of the mouse. Otherwise it rotates in the opposite direction of the mouse movement
	var cos_x = Math.cos(-rotationX);
	var sin_x = Math.sin(-rotationX);
	var cos_y = Math.cos(-rotationY);
	var sin_y = Math.sin(-rotationY);
	
	// [TO-DO] Modify the code below to form the transformation matrix.
	//I have applied the transformations in the following order: T*Ry*Rx
	var trans = [
    cos_y, sin_x*sin_y, sin_y*cos_x, 0,
    0, cos_x, -sin_x, 0,
    -sin_y, sin_x*cos_y, cos_x * cos_y, 0,
    translationX, translationY, translationZ, 1
];
	var mv = trans;
	return mv;
}


// [TO-DO] Complete the implementation of the following class.

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		 // [TO-DO] initializations
        // Initialize WebGL context (already done in the .html file)
        // Create the vertex shader
		const vs_source = `
			attribute vec3 pos;
			attribute vec2 txc;
			attribute vec3 normal;
			
			uniform mat4 mvp;
			uniform mat4 mv;
			uniform mat3 normalMat;
			uniform vec3 lightDir;
			
			varying vec2 vtexCoord;
			varying vec3 vNormal;

			uniform bool swapYZ;

			void main() {
				vec3 position = pos;
				if (swapYZ) {
					position = vec3(pos.x, pos.z, pos.y);
				}
				gl_Position = mvp * vec4(position, 1.0);
				vtexCoord = txc;

				// Transform the normal to eye space and normalize
				vNormal = normalize(normalMat * normal);
			}
		`;
        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vs_source);
        gl.compileShader(vs);

        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(vs));
            gl.deleteShader(vs);
        }

        // Create the fragment shader
		const fs_source = `
			precision highp float;
			uniform sampler2D tex;
			uniform bool showTex;
			uniform vec3 lightDir;
			uniform vec3 ambientLightColor;
			
			varying vec2 vtexCoord;
			varying vec3 vNormal;
			
			uniform mat4 mv;
			
			uniform float shininess;

			void main() {
				vec3 diffuseColor = vec3(0.8); // Diffuse color
				vec3 specularColor = vec3(1.0); // Specular color
				vec3 ambientLightColor = vec3(0.3); // Ambient color

				vec3 texColor = vec3(1.0); // Default color without texture
				if (showTex) {
					texColor = texture2D(tex, vtexCoord).rgb;
				}
				
				// Ambient light term
				vec3 ambient = ambientLightColor * texColor;

				// Diffuse term
				vec3 normal = normalize(vNormal);
				vec3 light = normalize(lightDir);
				float NdotL = max(dot(normal, light), 0.0);
				vec3 diffuse = diffuseColor * texColor * NdotL;
				
				// Shadow term 
				vec3 shadowColor = vec3(0.2); 
				float shadowFactor = 1.0 - NdotL;
				diffuse = mix(diffuse, shadowColor, shadowFactor);
				
				// Specular term
				vec3 viewDir = normalize(-mv[3].xyz);
				vec3 halfwayDir = normalize(lightDir + viewDir);
				float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
				vec3 specular = specularColor * spec;

				vec3 finalColor = ambient + diffuse + specular;

				gl_FragColor = vec4(finalColor, 1.0);
			}
		`;

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fs_source);
        gl.compileShader(fs);

        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fs));
            gl.deleteShader(fs);
        }

        this.prog = gl.createProgram();
        gl.attachShader(this.prog, vs);
        gl.attachShader(this.prog, fs);
        gl.linkProgram(this.prog);

        if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
            alert(gl.getProgramInfoLog(this.prog));

        }

        this.position_buffer = gl.createBuffer();
        this.texture_buffer = gl.createBuffer();
        this.normal_buffer = gl.createBuffer();
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		// [TO-DO] Update the contents of the vertex buffer objects.
        this.numTriangles = vertPos.length / 3;

        //Passing position variables from CPU to GPU
        gl.bindBuffer(gl.ARRAY_BUFFER, this.position_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);

        //Passing texture variables from CPU to GPU
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texture_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);

        //Passing normal variables from CPU to GPU
        gl.bindBuffer(gl.ARRAY_BUFFER, this.normal_buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		// [TO-DO] Set the uniform parameter(s) of the vertex shader
        gl.useProgram(this.prog);
        const swapYZLocation = gl.getUniformLocation(this.prog, 'swapYZ');
        gl.uniform1i(swapYZLocation, swap ? 1 : 0);
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw( matrixMVP, matrixMV, matrixNormal )
	{
		// [TO-DO] Complete the WebGL initializations before drawing
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(this.prog);

        const mvpLocation = gl.getUniformLocation(this.prog, 'mvp');
        gl.uniformMatrix4fv(mvpLocation, false, matrixMVP);

        const mvLocation = gl.getUniformLocation(this.prog, 'mv');
        gl.uniformMatrix4fv(mvLocation, false, matrixMV);

        const normalMatLocation = gl.getUniformLocation(this.prog, 'normalMat');
        gl.uniformMatrix3fv(normalMatLocation, false, matrixNormal);

        // Activate position buffer
        const posAttribLocation = gl.getAttribLocation(this.prog, 'pos');
        gl.enableVertexAttribArray(posAttribLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.position_buffer);
        gl.vertexAttribPointer(posAttribLocation, 3, gl.FLOAT, false, 0, 0);

        // Activate texture coordinates buffer
        const texCoordAttribLocation = gl.getAttribLocation(this.prog, 'txc');
        gl.enableVertexAttribArray(texCoordAttribLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.texture_buffer);
        gl.vertexAttribPointer(texCoordAttribLocation, 2, gl.FLOAT, false, 0, 0);

        // Activate normal buffer
        const normalAttribLocation = gl.getAttribLocation(this.prog, 'normal');
        gl.enableVertexAttribArray(normalAttribLocation);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.normal_buffer);
        gl.vertexAttribPointer(normalAttribLocation, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, this.numTriangles);
	}
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture(img) {
		const mytex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, mytex);
		
		// Set the texture image data
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
		gl.generateMipmap(gl.TEXTURE_2D);
		
		// Set texture parameters
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		
		// Assign the texture unit 0 to the sampler uniform
		const sampler = gl.getUniformLocation(this.prog, 'tex');
		gl.useProgram(this.prog);
		gl.uniform1i(sampler, 0);
	}

	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		 // [TO-DO] set the uniform parameter(s) of the fragment shader to specify if it should use the texture.
        const showTexLocation = gl.getUniformLocation(this.prog, 'showTex');
        gl.useProgram(this.prog);
        gl.uniform1i(showTexLocation, show ? 1 : 0);
	}
	
	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
		gl.useProgram(meshDrawer.prog);
		const lightDirLocation = gl.getUniformLocation(meshDrawer.prog, 'lightDir');
		gl.uniform3f(lightDirLocation, x, y, z);
	}
	
	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		 gl.useProgram(meshDrawer.prog);
		const shininessLocation = gl.getUniformLocation(meshDrawer.prog, 'shininess');
		gl.uniform1f(shininessLocation, shininess);
	}
}


// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep(dt, positions, velocities, springs, stiffness, damping, gravity, restitution, substeps, pinnedParticles, holdVert, selVert) {
    var boxSize = 1.0;

    // Backup old positions
    var old_positions = positions.map(p => new Vec3(p.x, p.y, p.z));

    // Apply gravity and initial velocity update
    for (var i = 0; i < positions.length; i++) {
        if (pinnedParticles.includes(i) || i === selVert) continue; // Skip pinned particles and held vertex
        velocities[i] = velocities[i].add(gravity.mul(dt));
        positions[i] = positions[i].add(velocities[i].mul(dt));
    }

    // Iterate to satisfy constraints (PBD)
    for (var k = 0; k < substeps; k++) {
        for (var i = 0; i < springs.length; i++) {
            var spring = springs[i];
            var p0 = spring.p0;
            var p1 = spring.p1;

            // Calculate the difference vector between the positions of the two particles
            var d = positions[p1].sub(positions[p0]);

            // Calculate the current length of the spring
            var distance = d.len();

            // Calculate the correction vector
            var correction = d.mul((distance - spring.rest) * stiffness / distance / 2);

            // Apply the correction to the particles
            if (!pinnedParticles.includes(p0) && p0 !== selVert) positions[p0] = positions[p0].add(correction);
            if (!pinnedParticles.includes(p1) && p1 !== selVert) positions[p1] = positions[p1].sub(correction);
        }
    }

    // Handle collisions with the box walls
    for (var i = 0; i < positions.length; i++) {
        // x-axis
        if (positions[i].x < -boxSize) {
            positions[i].x = -boxSize;
            velocities[i].x *= -restitution;
        } else if (positions[i].x > boxSize) {
            positions[i].x = boxSize;
            velocities[i].x *= -restitution;
        }
        // y-axis
        if (positions[i].y < -boxSize) {
            positions[i].y = -boxSize;
            velocities[i].y *= -restitution;
        } else if (positions[i].y > boxSize) {
            positions[i].y = boxSize;
            velocities[i].y *= -restitution;
        }
        // z-axis
        if (positions[i].z < -boxSize) {
            positions[i].z = -boxSize;
            velocities[i].z *= -restitution;
        } else if (positions[i].z > boxSize) {
            positions[i].z = boxSize;
            velocities[i].z *= -restitution;
        }
    }

    // Update velocities
    for (var i = 0; i < positions.length; i++) {
        if (pinnedParticles.includes(i) || i === selVert) continue; // Skip pinned particles and held vertex
        velocities[i] = positions[i].sub(old_positions[i]).div(dt).mul(1 - damping);
    }

    // Ensure held vertex stays in place
    if (holdVert && selVert !== undefined) {
        positions[selVert] = holdVert.copy();
        velocities[selVert].init(0, 0, 0);
    }
}


