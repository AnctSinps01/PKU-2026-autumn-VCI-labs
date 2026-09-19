#include <random>

#include <spdlog/spdlog.h>

#include "Labs/1-Drawing2D/tasks.h"

using VCX::Labs::Common::ImageRGB;

namespace VCX::Labs::Drawing2D {
    /******************* 1.Image Dithering *****************/
    void DitheringThreshold(
        ImageRGB &       output,
        ImageRGB const & input) {
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringRandomUniform(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        static std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                color += glm::vec3(dist(gen));
                output.At(x, y) = {
                    color.r > 0.5 ? 1 : 0,
                    color.g > 0.5 ? 1 : 0,
                    color.b > 0.5 ? 1 : 0,
                };
            }
    }

    void DitheringRandomBlueNoise(
        ImageRGB &       output,
        ImageRGB const & input,
        ImageRGB const & noise) {
        // your code here:
        for (std::size_t x = 0; x < input.GetSizeX(); ++x)
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::vec3 color = input.At(x, y);
                glm::vec3 thres = noise.At(x, y);
                output.At(x, y) = {
                    color.r + (thres.r - .5f) > .5f,
                    color.g + (thres.g - .5f) > .5f,
                    color.b + (thres.b - .5f) > .5f
                };
            }
    }

    void DitheringOrdered(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        static const int lines[3][30] = {
            {0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1},
            {0,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,1, 0,1,1, 1,1,1},
            {0,0,0, 0,0,0, 1,0,0, 1,0,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1}
        };
        for (std::size_t x = 0; x < input.GetSizeX(); ++x) {
            for (std::size_t y = 0; y < input.GetSizeY(); ++y) {
                glm::ivec3 indices(glm::round(input.At(x, y) * 9.f));
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        output.At(x *3 + i, y *3 + j) = {
                            lines[i][indices.r *3 + j],
                            lines[i][indices.g *3 + j],
                            lines[i][indices.b *3 + j],
                        };
            }
        }
    }

    void DitheringErrorDiffuse(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        int width  = int(input.GetSizeX());
        int height = int(input.GetSizeY());
        std::vector<glm::vec3> buffer(width * height);
        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y)
                buffer[y * width + x] = input.At(x, y);

        for (int y = 0; y < height; ++y) {
            int dir = y % 2 == 0 ? 1 : -1;
            for (int i = 0; i < width; ++i) {
                int x = dir > 0 ? i : width - 1 - i;
                glm::vec3 color = buffer[y * width + x];
                glm::vec3 newColor = {
                    color.r > 0.5f ? 1.f : 0.f,
                    color.g > 0.5f ? 1.f : 0.f,
                    color.b > 0.5f ? 1.f : 0.f,
                };
                output.At(x, y) = newColor;
                glm::vec3 err = color - newColor;
                if (x + dir >= 0 && x + dir < width)
                    buffer[y * width + x + dir] += err * (7.f / 16.f);
                if (y + 1 < height) {
                    if (x - dir >= 0 && x - dir < width)
                        buffer[(y + 1) * width + x - dir] += err * (3.f / 16.f);
                    buffer[(y + 1) * width + x] += err * (5.f / 16.f);
                    if (x + dir >= 0 && x + dir < width)
                        buffer[(y + 1) * width + x + dir] += err * (1.f / 16.f);
                }
            }
        }
    }

    /******************* 2.Image Filtering *****************/
    void Blur(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        int width  = input.GetSizeX();
        int height = input.GetSizeY();
        std::vector<glm::vec3> temp(width * height);

        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y) {
                glm::vec3 color = input.At(x, y) * 2.f;
                float count = 2.f;
                if (y > 0) { color += input.At(x, y - 1); count += 1.f; }
                if (y + 1 < height) { color += input.At(x, y + 1); count += 1.f; }
                temp[y * width + x] = color / count;
            }

        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y) {
                glm::vec3 color = temp[y * width + x] * 2.f;
                float count = 2.f;
                if (x > 0) { color += temp[y * width + x - 1]; count += 1.f; }
                if (x + 1 < width) { color += temp[y * width + x + 1]; count += 1.f; }
                output.At(x, y) = color / count;
            }
    }

    void Edge(
        ImageRGB &       output,
        ImageRGB const & input) {
        // your code here:
        int width  = input.GetSizeX();
        int height = input.GetSizeY();
        std::vector<glm::vec3> smoothX(width * height);
        std::vector<glm::vec3> smoothY(width * height);

        for (int x = 0; x < width; ++x)
            for (int y = 1; y + 1 < height; ++y)
                smoothY[y * width + x] = input.At(x, y - 1) + input.At(x, y) * 2.f + input.At(x, y + 1);

        for (int x = 1; x + 1 < width; ++x)
            for (int y = 0; y < height; ++y)
                smoothX[y * width + x] = input.At(x - 1, y) + input.At(x, y) * 2.f + input.At(x + 1, y);

        for (int x = 1; x + 1 < width; ++x)
            for (int y = 1; y + 1 < height; ++y) {
                glm::vec3 gx = smoothY[y * width + x + 1] - smoothY[y * width + x - 1];
                glm::vec3 gy = smoothX[(y + 1) * width + x] - smoothX[(y - 1) * width + x];
                output.At(x, y) = glm::sqrt(gx * gx + gy * gy);
            }
    }

    /******************* 3. Image Inpainting *****************/
    void Inpainting(
        ImageRGB &         output,
        ImageRGB const &   inputBack,
        ImageRGB const &   inputFront,
        const glm::ivec2 & offset) {
        output             = inputBack;
        std::size_t width  = inputFront.GetSizeX();
        std::size_t height = inputFront.GetSizeY();
        glm::vec3 * g      = new glm::vec3[width * height];
        memset(g, 0, sizeof(glm::vec3) * width * height);
        // set boundary condition
        for (std::size_t y = 0; y < height; ++y) {
            // set boundary for (0, y), your code: g[y * width] = ?
            g[y * width] = inputBack.At(offset.x, offset.y + y) - inputFront.At(0, y);
            // set boundary for (width - 1, y), your code: g[y * width + width - 1] = ?
            g[y * width + width - 1] = 
                inputBack.At(offset.x + width - 1, offset.y + y) - inputFront.At(width - 1, y);
        }
        for (std::size_t x = 0; x < width; ++x) {
            // set boundary for (x, 0), your code: g[x] = ?
            g[x] = inputBack.At(offset.x + x, offset.y) - inputFront.At(x, 0);
            // set boundary for (x, height - 1), your code: g[(height - 1) * width + x] = ?
            g[(height - 1) * width + x] =
                inputBack.At(offset.x + x, offset.y + height - 1) - inputFront.At(x, height - 1);
        }

        // Jacobi iteration, solve Ag = b
        for (int iter = 0; iter < 8000; ++iter) {
            for (std::size_t y = 1; y < height - 1; ++y)
                for (std::size_t x = 1; x < width - 1; ++x) {
                    g[y * width + x] = (g[(y - 1) * width + x] + g[(y + 1) * width + x] + g[y * width + x - 1] + g[y * width + x + 1]);
                    g[y * width + x] = g[y * width + x] * glm::vec3(0.25);
                }
        }

        for (std::size_t y = 0; y < inputFront.GetSizeY(); ++y)
            for (std::size_t x = 0; x < inputFront.GetSizeX(); ++x) {
                glm::vec3 color = g[y * width + x] + inputFront.At(x, y);
                output.At(x + offset.x, y + offset.y) = color;
            }
        delete[] g;
    }

    /******************* 4. Line Drawing *****************/
    void DrawLine(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1) {
        // your code here:
        glm::ivec2 dif = glm::abs(p1 - p0);
        glm::ivec2 del = glm::sign(p1 - p0);
        int err = dif.x - dif.y; 
        glm::ivec2 curr = p0;
        while (1) {
            canvas.At(curr.x, curr.y) = color;
            if (curr == p1) break;
            int er = 2 * err;
            if (er > -dif.y) {
                err -= dif.y;
                curr.x += del.x;
            }
            if (er < dif.x) {
                err += dif.x;
                curr.y += del.y;
            }
        }
    }

    /******************* 5. Triangle Drawing *****************/
    void DrawTriangleFilled(
        ImageRGB &       canvas,
        glm::vec3 const  color,
        glm::ivec2 const p0,
        glm::ivec2 const p1,
        glm::ivec2 const p2) {
        // your code here:
        static auto cross = [&] (const glm::vec2 &a, const glm::vec2& b) -> float {
            return a.x * b.y - a.y * b.x;
        };
        glm::vec2 ab = p1 - p0;
        glm::vec2 ac = p2 - p0;
        float det = cross(ab, ac);
        if (std::abs(det) < 1.f) return;
        glm::ivec2 const SIZE = {canvas.GetSizeX() -1, canvas.GetSizeY() -1};
        glm::ivec2 box_min = glm::min(p0, glm::min(p1, p2));
        glm::ivec2 box_max = glm::max(p0, glm::max(p1, p2));
        box_min = glm::clamp(box_min, glm::ivec2(0), SIZE);
        box_max = glm::clamp(box_max, glm::ivec2(0), SIZE);
        glm::ivec2 curr = box_min;
        ab /= det;
        ac /= det;
        for (; curr.x <= box_max.x; ++curr.x) {
            for (; curr.y <= box_max.y; ++curr.y) {
                glm::vec2 ap = curr - p0;
                float b = cross(ab, ap);
                float c = cross(ap, ac);
                if (b >= 0 && c >= 0 && b + c <= 1.f) canvas.At(curr.x, curr.y) = color;
            }
            curr.y = box_min.y;
        }
    }

    /******************* 6. Image Supersampling *****************/
    void Supersample(
        ImageRGB &       output,
        ImageRGB const & input,
        int              rate) {
        // your code here:
        auto sample = [&input] (float x, float y) -> glm::vec3 {
            glm::vec2  uv(x - 0.5, y - 0.5);
            glm::ivec2 p0 = glm::ivec2(glm::floor(uv));
            glm::vec2  t  = glm::fract(uv);
            auto texel = [&input] (int x, int y) -> glm::vec3 {
                return input.At(
                    std::size_t(std::clamp(x, 0, int(input.GetSizeX()) - 1)),
                    std::size_t(std::clamp(y, 0, int(input.GetSizeY()) - 1)));
            };
            return glm::mix(
                glm::mix(texel(p0.x, p0.y), texel(p0.x +1, p0.y), t.x),
                glm::mix(texel(p0.x, p0.y +1), texel(p0.x +1, p0.y +1), t.x),
                t.y);
        };
        ImageRGB Temp(output.GetSizeX(), output.GetSizeY() * rate);
        float stepX = float(input.GetSizeX()) / float(output.GetSizeX() * rate);
        float stepX_2 = stepX * .5f;
        float stepY = float(input.GetSizeY()) / float(output.GetSizeY() * rate);
        float stepY_2 = stepY * .5f;
        for (int y = 0; y < output.GetSizeY() * rate; ++y)
            for (int x = 0; x < output.GetSizeX(); ++x) {
                glm::vec3 tp(0.f);
                for (int i = 0; i < rate; ++i)
                    tp += sample(stepX * (x * rate + i) + stepX_2, stepY * y + stepY_2);
                tp /= float(rate);
                Temp.At(x, y) = tp;
            }
        for (int x = 0; x < output.GetSizeX(); ++x)
            for (int y = 0; y < output.GetSizeY(); ++y) {
                glm::vec3 tp(0.f);
                for (int i = 0; i < rate; ++i)
                    tp += glm::vec3(Temp.At(x, y * rate + i));
                tp /= float(rate);
                output.At(x, y) = tp;
            }
    }

    /******************* 7. Bezier Curve *****************/
    // Note: Please finish the function [DrawLine] before trying this part.
    glm::vec2 CalculateBezierPoint(
        std::span<glm::vec2> points,
        float const          t) {
        // your code here:
        int N = points.size();
        std::vector<glm::vec2> ps(points.begin(), points.end());
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N -i -1; ++j)
                ps[j] = glm::mix(ps[j], ps[j +1], t);
        return ps[0];
    }
} // namespace VCX::Labs::Drawing2D