/**
 * WebGL2 GPU-Accelerated PDE Solver
 * Implements: ∂ₜy = M(κΔy + λ(K*y) - α(y-y₀)³)
 */

class GPUSolver {
    constructor(size = 256) {
        this.N = size;
        this.canvas = document.createElement('canvas');
        this.gl = this.canvas.getContext('webgl2', {
            antialias: false,
            depth: false,
            alpha: false,
            premultipliedAlpha: false,
            preserveDrawingBuffer: true
        });

        if (!this.gl) {
            console.warn('WebGL2 not available, falling back to CPU');
            this.gpuAvailable = false;
            return;
        }

        this.gpuAvailable = true;
        this.initGL();
        this.compileShaders();
        this.createTextures();
        this.createFramebuffers();
    }

    initGL() {
        const gl = this.gl;
        this.canvas.width = this.N;
        this.canvas.height = this.N;

        // Quad for full-screen rendering
        const vertices = new Float32Array([
            -1, -1,  1, -1,  -1, 1,
            -1,  1,  1, -1,   1, 1
        ]);

        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    }

    compileShaders() {
        const gl = this.gl;

        // Vertex shader
        const vertSource = `#version 300 es
            in vec2 a_position;
            out vec2 v_texCoord;
            void main() {
                v_texCoord = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        // Fragment shader: PDE time step
        const fragSource = `#version 300 es
            precision highp float;
            uniform sampler2D u_field;
            uniform sampler2D u_kernel;
            uniform vec2 u_resolution;
            uniform float u_kappa;
            uniform float u_lambda;
            uniform float u_alpha;
            uniform float u_M;
            uniform float u_y0;
            uniform float u_dt;
            uniform int u_kernelSize;
            in vec2 v_texCoord;
            out vec4 fragColor;

            // Compute Laplacian using finite differences
            float laplacian(vec2 uv) {
                vec2 texel = 1.0 / u_resolution;
                float center = texture(u_field, uv).r;
                float left   = texture(u_field, uv + vec2(-texel.x, 0.0)).r;
                float right  = texture(u_field, uv + vec2( texel.x, 0.0)).r;
                float up     = texture(u_field, uv + vec2(0.0,  texel.y)).r;
                float down   = texture(u_field, uv + vec2(0.0, -texel.y)).r;

                return (left + right + up + down - 4.0 * center) * 100.0;
            }

            // Convolution with kernel
            float convolve(vec2 uv) {
                float sum = 0.0;
                int halfK = u_kernelSize / 2;
                vec2 texel = 1.0 / u_resolution;

                for (int i = -halfK; i <= halfK; i++) {
                    for (int j = -halfK; j <= halfK; j++) {
                        vec2 offset = vec2(float(i), float(j)) * texel;
                        vec2 sampleUV = fract(uv + offset); // Periodic BC
                        float fieldVal = texture(u_field, sampleUV).r;

                        vec2 kernelUV = (vec2(float(i + halfK), float(j + halfK)) + 0.5) / float(u_kernelSize);
                        float kernelVal = texture(u_kernel, kernelUV).r;

                        sum += fieldVal * kernelVal;
                    }
                }
                return sum;
            }

            void main() {
                float y = texture(u_field, v_texCoord).r;
                float lap = laplacian(v_texCoord);
                float conv = convolve(v_texCoord);

                float dydt = u_M * (u_kappa * lap + u_lambda * conv - u_alpha * pow(y - u_y0, 3.0));
                float y_next = y + u_dt * dydt;

                fragColor = vec4(y_next, 0.0, 0.0, 1.0);
            }
        `;

        this.program = this.createProgram(vertSource, fragSource);

        // Get attribute and uniform locations
        this.locations = {
            position: gl.getAttribLocation(this.program, 'a_position'),
            field: gl.getUniformLocation(this.program, 'u_field'),
            kernel: gl.getUniformLocation(this.program, 'u_kernel'),
            resolution: gl.getUniformLocation(this.program, 'u_resolution'),
            kappa: gl.getUniformLocation(this.program, 'u_kappa'),
            lambda: gl.getUniformLocation(this.program, 'u_lambda'),
            alpha: gl.getUniformLocation(this.program, 'u_alpha'),
            M: gl.getUniformLocation(this.program, 'u_M'),
            y0: gl.getUniformLocation(this.program, 'u_y0'),
            dt: gl.getUniformLocation(this.program, 'u_dt'),
            kernelSize: gl.getUniformLocation(this.program, 'u_kernelSize')
        };
    }

    createProgram(vertSource, fragSource) {
        const gl = this.gl;

        const vertShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertShader, vertSource);
        gl.compileShader(vertShader);

        if (!gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
            console.error('Vertex shader error:', gl.getShaderInfoLog(vertShader));
        }

        const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragShader, fragSource);
        gl.compileShader(fragShader);

        if (!gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
            console.error('Fragment shader error:', gl.getShaderInfoLog(fragShader));
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertShader);
        gl.attachShader(program, fragShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
        }

        return program;
    }

    createTextures() {
        const gl = this.gl;

        // Ping-pong textures for field
        this.textures = {
            field: [this.createFloatTexture(), this.createFloatTexture()],
            kernel: this.createFloatTexture()
        };

        this.currentField = 0;
    }

    createFloatTexture() {
        const gl = this.gl;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.N, this.N, 0, gl.RED, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        return texture;
    }

    createFramebuffers() {
        const gl = this.gl;
        this.framebuffers = [gl.createFramebuffer(), gl.createFramebuffer()];

        for (let i = 0; i < 2; i++) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[i]);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                this.textures.field[i],
                0
            );
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    uploadField(data) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.textures.field[this.currentField]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.N, this.N, 0, gl.RED, gl.FLOAT, data);
    }

    uploadKernel(data, size) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.textures.kernel);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, size, size, 0, gl.RED, gl.FLOAT, data);
        this.kernelSize = size;
    }

    step(params) {
        if (!this.gpuAvailable) return;

        const gl = this.gl;
        const nextField = 1 - this.currentField;

        // Bind framebuffer for next field
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[nextField]);
        gl.viewport(0, 0, this.N, this.N);

        // Use shader program
        gl.useProgram(this.program);

        // Bind textures
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.field[this.currentField]);
        gl.uniform1i(this.locations.field, 0);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.kernel);
        gl.uniform1i(this.locations.kernel, 1);

        // Set uniforms
        gl.uniform2f(this.locations.resolution, this.N, this.N);
        gl.uniform1f(this.locations.kappa, params.kappa);
        gl.uniform1f(this.locations.lambda, params.lambda);
        gl.uniform1f(this.locations.alpha, params.alpha);
        gl.uniform1f(this.locations.M, params.M);
        gl.uniform1f(this.locations.y0, params.y0);
        gl.uniform1f(this.locations.dt, params.dt);
        gl.uniform1i(this.locations.kernelSize, this.kernelSize);

        // Bind quad buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.enableVertexAttribArray(this.locations.position);
        gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);

        // Draw
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Swap
        this.currentField = nextField;

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    readField() {
        const gl = this.gl;
        const data = new Float32Array(this.N * this.N);

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers[this.currentField]);
        gl.readPixels(0, 0, this.N, this.N, gl.RED, gl.FLOAT, data);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        return data;
    }
}

export { GPUSolver };
