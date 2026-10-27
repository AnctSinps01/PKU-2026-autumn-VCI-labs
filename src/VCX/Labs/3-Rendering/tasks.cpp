#include "Labs/3-Rendering/tasks.h"

namespace VCX::Labs::Rendering {

    glm::vec4 GetTexture(Engine::Texture2D<Engine::Formats::RGBA8> const & texture, glm::vec2 const & uvCoord) {
        if (texture.GetSizeX() == 1 || texture.GetSizeY() == 1) return texture.At(0, 0);
        glm::vec2 uv      = glm::fract(uvCoord);
        uv.x              = uv.x * texture.GetSizeX() - .5f;
        uv.y              = uv.y * texture.GetSizeY() - .5f;
        std::size_t xmin  = std::size_t(glm::floor(uv.x) + texture.GetSizeX()) % texture.GetSizeX();
        std::size_t ymin  = std::size_t(glm::floor(uv.y) + texture.GetSizeY()) % texture.GetSizeY();
        std::size_t xmax  = (xmin + 1) % texture.GetSizeX();
        std::size_t ymax  = (ymin + 1) % texture.GetSizeY();
        float       xfrac = glm::fract(uv.x), yfrac = glm::fract(uv.y);
        return glm::mix(glm::mix(texture.At(xmin, ymin), texture.At(xmin, ymax), yfrac), glm::mix(texture.At(xmax, ymin), texture.At(xmax, ymax), yfrac), xfrac);
    }

    glm::vec4 GetAlbedo(Engine::Material const & material, glm::vec2 const & uvCoord) {
        glm::vec4 albedo       = GetTexture(material.Albedo, uvCoord);
        glm::vec3 diffuseColor = albedo;
        return glm::vec4(glm::pow(diffuseColor, glm::vec3(2.2)), albedo.w);
    }

    /******************* 1. Ray-triangle intersection *****************/
    bool IntersectTriangle(Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3) {
        glm::vec3 e1 = p2 - p1;
        glm::vec3 e2 = p3 - p1;
        glm::vec3 s  = ray.Origin - p1;
        glm::vec3 p  = glm::cross(ray.Direction, e2);
        float     det = glm::dot(e1, p);
        if (glm::abs(det) < EPS2) return false;

        float invDet = 1.0f / det;
        float u      = glm::dot(s, p) * invDet;
        if (u < 0.0f || u > 1.0f) return false;

        glm::vec3 q = glm::cross(s, e1);
        float     v = glm::dot(ray.Direction, q) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;

        float t = glm::dot(e2, q) * invDet;
        if (t < 0.0f) return false;

        output.t = t;
        output.u = u;
        output.v = v;
        return true;
    }

    glm::vec3 RayTrace(const RayIntersector & intersector, Ray ray, int maxDepth, bool enableShadow) {
        glm::vec3 color(0.0f);
        glm::vec3 weight(1.0f);

        for (int depth = 0; depth < maxDepth; depth++) {
            auto rayHit = intersector.IntersectRay(ray);
            if (! rayHit.IntersectState) return color;
            const glm::vec3 pos       = rayHit.IntersectPosition;
            const glm::vec3 n         = glm::normalize(rayHit.IntersectNormal);
            const glm::vec3 kd        = rayHit.IntersectAlbedo;
            const glm::vec3 ks        = rayHit.IntersectMetaSpec;
            const float     alpha     = rayHit.IntersectAlbedo.w;
            const float     shininess = rayHit.IntersectMetaSpec.w * 256;

            glm::vec3 result = kd * intersector.InternalScene->AmbientIntensity;
            /******************* 2. Whitted-style ray tracing *****************/
            for (const Engine::Light & light : intersector.InternalScene->Lights) {
                glm::vec3 l;
                float     attenuation;
                /******************* 3. Shadow ray *****************/
                if (light.Type == Engine::LightType::Point) {
                    l           = light.Position - pos;
                    attenuation = 1.0f / glm::dot(l, l);
                    if (enableShadow) {
                        float dist = glm::length(l);
                        auto  hit  = intersector.IntersectRay(Ray(pos, l / dist));
                        if (hit.IntersectState && hit.IntersectAlbedo.w >= 0.2f && glm::length(hit.IntersectPosition - pos) < dist) continue;
                    }
                } else if (light.Type == Engine::LightType::Directional) {
                    l           = light.Direction;
                    attenuation = 1.0f;
                    if (enableShadow) {
                        auto hit = intersector.IntersectRay(Ray(pos, l));
                        if (hit.IntersectState && hit.IntersectAlbedo.w >= 0.2f) continue;
                    }
                }

                /******************* 2. Whitted-style ray tracing *****************/
                l           = glm::normalize(l);
                float diff  = glm::max(glm::dot(l, n), 0.0f);
                float spec  = glm::pow(glm::max(glm::dot(glm::reflect(-l, n), -ray.Direction), 0.0f), shininess);
                result     += light.Intensity * attenuation * (kd * diff + ks * spec);
            }

            if (alpha < 0.9) {
                // refraction
                // accumulate color
                glm::vec3 R = alpha * glm::vec3(1.0f);
                color += weight * R * result;
                weight *= glm::vec3(1.0f) - R;

                // generate new ray
                ray = Ray(pos, ray.Direction);
            } else {
                // reflection
                // accumulate color
                glm::vec3 R = ks * glm::vec3(0.5f);
                color += weight * (glm::vec3(1.0f) - R) * result;
                weight *= R;

                // generate new ray
                glm::vec3 out_dir = ray.Direction - glm::vec3(2.0f) * n * glm::dot(n, ray.Direction);
                ray               = Ray(pos, out_dir);
            }
        }

        return color;
    }
} // namespace VCX::Labs::Rendering