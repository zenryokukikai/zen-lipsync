// Matte output (cut-out avatar): the idle loop is a stacked video, the top half the frame with the background's colour
// taken out of the hair and edges, the bottom half the person's matte (grey). The player composes the presented frame
// and its mouth patch on its (hidden) 2D canvas as in 'canvas' sync (the whole stacked frame: the patch lands in the top
// half), and this draws that canvas to a WebGL canvas in the same frame callback: colour from the top half, alpha from
// the bottom half (the drawing buffer is kept: the page may read the canvas between frames). Which frame is drawn is the player's frame callback's, so the cut-out follows the idle video exactly.

const VS = `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = vec2((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`;
const FS = `#version 300 es
precision mediump float;
uniform sampler2D t; in vec2 uv; out vec4 o;
void main() {
  vec3 c = texture(t, vec2(uv.x, uv.y * 0.5)).rgb;
  float a = texture(t, vec2(uv.x, 0.5 + uv.y * 0.5)).g;
  o = vec4(c * a, a);
}`;

export class MatteOutput {
  private readonly gl: WebGL2RenderingContext;
  private readonly tex: WebGLTexture;

  /** out: the visible canvas (frame size); source: the player's 2D canvas (frame width x twice the frame height). */
  constructor(readonly out: HTMLCanvasElement, private readonly source: HTMLCanvasElement) {
    const gl = out.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('ZEN-LipSync: the matte output needs WebGL2');
    this.gl = gl;
    const sh = (type: number, src: string): WebGLShader => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`ZEN-LipSync: matte shader: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`ZEN-LipSync: matte program: ${gl.getProgramInfoLog(prog)}`);
    gl.useProgram(prog);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  }

  /** Draws the player's composed canvas (call right after the player composed the presented frame). */
  render(): void {
    const gl = this.gl, s = this.source;
    if (this.out.width !== s.width || this.out.height !== s.height / 2) { this.out.width = s.width; this.out.height = s.height / 2; }
    gl.viewport(0, 0, this.out.width, this.out.height);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, s);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  destroy(): void { this.gl.getExtension('WEBGL_lose_context')?.loseContext(); }
}
