#version 140

// Bambu Colorin (Orca FullSpectrum 0.9.18): per-vertex colour preview of an OBJ image map.
uniform vec4 uniform_color;

in vec2 intensity;
in vec3 color;

out vec4 out_color;

void main()
{
    out_color = vec4(vec3(intensity.y) + color * intensity.x, uniform_color.a);
}
