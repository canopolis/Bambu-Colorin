#version 110

// Bambu Colorin (Orca FullSpectrum 0.9.18): per-vertex colour preview of an OBJ image map.
uniform vec4 uniform_color;

varying vec2 intensity;
varying vec3 color;

void main()
{
    gl_FragColor = vec4(vec3(intensity.y) + color * intensity.x, uniform_color.a);
}
