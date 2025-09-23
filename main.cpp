/**********************************************************************************
 * ModelConverter.cpp
 *
 *
 *
 * LI WENHUI
 * 2025/09/20
 **********************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/quaternion.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// 模型缩放
const float G_SCALE_FACTOR = 0.01f;

struct Vertex {
    float position[3]{};
    float texcoord[2]{};
    float normal[3]{};
    float tangent[3]{};
    int   boneIDs[4]{ -1,-1,-1,-1 };
    float weights[4]{ 0,0,0,0 };
};

struct MeshHeader {
    uint32_t vertexCount{};
    uint32_t indexCount{};
    uint32_t materialIndex{};
};

struct TempBoneInfo {
    std::string name;
    unsigned int originalIndex;
    int parentIndex;
    aiMatrix4x4 offsetMatrix;
};

static inline void AddBoneWeight(Vertex& v, int boneId, float w) {
    for (int i = 0; i < 4; ++i) if (v.boneIDs[i] < 0) { v.boneIDs[i] = boneId; v.weights[i] = w; return; }
    int minIdx = 0; for (int i = 1; i < 4; ++i) if (v.weights[i] < v.weights[minIdx]) minIdx = i;
    if (w > v.weights[minIdx]) { v.boneIDs[minIdx] = boneId; v.weights[minIdx] = w; }
}

static inline void NormalizeWeights(Vertex& v) {
    float s = v.weights[0] + v.weights[1] + v.weights[2] + v.weights[3];
    if (s > 1e-6f) { float inv = 1.0f / s; for (float& x : v.weights) x *= inv; }
}

static inline json MatrixToJson(const aiMatrix4x4& m) {
    return json::array({ m.a1, m.b1, m.c1, m.d1,
                         m.a2, m.b2, m.c2, m.d2,
                         m.a3, m.b3, m.c3, m.d3,
                         m.a4, m.b4, m.c4, m.d4 });
}

static void BuildNodeMap(const aiNode* n, std::unordered_map<std::string, const aiNode*>& map) {
    if (!n) return; map[n->mName.C_Str()] = n; for (unsigned i = 0; i < n->mNumChildren; ++i) BuildNodeMap(n->mChildren[i], map);
}

static bool FindBoneOffset(const aiScene* s, const std::string& name, aiMatrix4x4& out) {
    for (unsigned m = 0; m < s->mNumMeshes; ++m) {
        const aiMesh* mesh = s->mMeshes[m];
        for (unsigned bi = 0; bi < mesh->mNumBones; ++bi) {
            const aiBone* b = mesh->mBones[bi];
            if (name == b->mName.C_Str()) { out = b->mOffsetMatrix; return true; }
        }
    }
    return false;
}

void processMesh(unsigned, const aiMesh*, const std::string&, const std::map<std::string, unsigned>&);
void processMaterial(unsigned int, const aiMaterial*, const aiScene*, const std::string&);
void processSkeleton(const aiScene*, const std::string&, std::map<std::string, unsigned>&, std::map<std::string, unsigned>&);
void processAnimation(unsigned, const aiAnimation*, const std::string&);
void createSceneFile(const aiScene*, const std::string&);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: ModelConverter.exe <输入文件.fbx>\n";
        return 1;
    }

    std::filesystem::path inPath(argv[1]);
    if (!std::filesystem::exists(inPath)) { std::cerr << "错误: 文件不存在: " << inPath << "\n"; return 1; }

    std::filesystem::path abs = std::filesystem::absolute(inPath);
    std::string outDir = inPath.stem().string();
    std::filesystem::create_directories(outDir);

    auto logln = [](const std::string& s) { std::cout << s << std::endl; };

    logln("[Info] Input : " + abs.string());
    logln("[Info] Output: " + std::filesystem::absolute(outDir).string());

    Assimp::Importer importer;
    unsigned flags = aiProcess_Triangulate | aiProcess_ConvertToLeftHanded | aiProcess_GenNormals | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality | aiProcess_OptimizeMeshes | aiProcess_SortByPType | aiProcess_CalcTangentSpace;
    const aiScene* scene = importer.ReadFile(abs.u8string(), flags);
    if (!scene) { logln(std::string("[Error] Assimp: ") + importer.GetErrorString()); return 1; }

    std::map<std::string, unsigned> tempBoneMap;
    unsigned tempBoneCounter = 0;
    for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[i];
        for (unsigned bi = 0; bi < mesh->mNumBones; ++bi) {
            std::string name = mesh->mBones[bi]->mName.C_Str();
            if (tempBoneMap.find(name) == tempBoneMap.end()) {
                tempBoneMap[name] = tempBoneCounter++;
            }
        }
    }

    std::map<std::string, unsigned> finalBoneMap;
    processSkeleton(scene, outDir, tempBoneMap, finalBoneMap);

    for (unsigned i = 0; i < scene->mNumMeshes; ++i)
        processMesh(i, scene->mMeshes[i], outDir, finalBoneMap);

    for (unsigned i = 0; i < scene->mNumMaterials; ++i)
        processMaterial(i, scene->mMaterials[i], scene, outDir);

    for (unsigned i = 0; i < scene->mNumAnimations; ++i)
        processAnimation(i, scene->mAnimations[i], outDir);

    createSceneFile(scene, outDir);

    logln("模型已成功拆分到目录: " + outDir);
    return 0;
}

void processMesh(unsigned idx, const aiMesh* mesh, const std::string& outDir, const std::map<std::string, unsigned>& finalBoneMap) {
    std::vector<Vertex> vertices(mesh->mNumVertices);
    for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
        vertices[i].position[0] = mesh->mVertices[i].x * G_SCALE_FACTOR;
        vertices[i].position[1] = mesh->mVertices[i].y * G_SCALE_FACTOR;
        vertices[i].position[2] = mesh->mVertices[i].z * G_SCALE_FACTOR;
        if (mesh->HasTextureCoords(0)) {
            vertices[i].texcoord[0] = mesh->mTextureCoords[0][i].x;
            vertices[i].texcoord[1] = mesh->mTextureCoords[0][i].y;
        }
        if (mesh->HasNormals()) {
            vertices[i].normal[0] = mesh->mNormals[i].x;
            vertices[i].normal[1] = mesh->mNormals[i].y;
            vertices[i].normal[2] = mesh->mNormals[i].z;
        }
        if (mesh->HasTangentsAndBitangents()) {
            vertices[i].tangent[0] = mesh->mTangents[i].x;
            vertices[i].tangent[1] = mesh->mTangents[i].y;
            vertices[i].tangent[2] = mesh->mTangents[i].z;
        }
    }
    for (unsigned bi = 0; bi < mesh->mNumBones; ++bi) {
        aiBone* b = mesh->mBones[bi];
        auto it = finalBoneMap.find(b->mName.C_Str());
        if (it != finalBoneMap.end()) {
            unsigned finalId = it->second;
            for (unsigned wi = 0; wi < b->mNumWeights; ++wi) {
                AddBoneWeight(vertices[b->mWeights[wi].mVertexId], (int)finalId, b->mWeights[wi].mWeight);
            }
        }
    }
    for (auto& v : vertices) NormalizeWeights(v);
    std::vector<uint32_t> indices;
    indices.reserve(mesh->mNumFaces * 3);
    for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
        for (unsigned j = 0; j < mesh->mFaces[f].mNumIndices; ++j) indices.push_back(mesh->mFaces[f].mIndices[j]);
    }
    std::string path = outDir + "/mesh_" + std::to_string(idx) + ".mesh";
    std::ofstream out(path, std::ios::binary);
    MeshHeader header{ (uint32_t)vertices.size(), (uint32_t)indices.size(), mesh->mMaterialIndex };
    out.write((char*)&header, sizeof(header));
    out.write((char*)vertices.data(), vertices.size() * sizeof(Vertex));
    out.write((char*)indices.data(), indices.size() * sizeof(uint32_t));
}

void processSkeleton(const aiScene* scene, const std::string& outDir, std::map<std::string, unsigned>& boneMap, std::map<std::string, unsigned>& finalBoneMap) {
    if (boneMap.empty()) {
        json j;
        j["bones"] = json::array();
        std::ofstream out(outDir + "/skeleton.json");
        out << j.dump(2);
        return;
    }
    std::unordered_map<std::string, const aiNode*> nodeMap;
    BuildNodeMap(scene->mRootNode, nodeMap);
    std::vector<TempBoneInfo> unsortedBones;
    for (const auto& kv : boneMap) {
        int parentId = -1;
        auto itN = nodeMap.find(kv.first);
        if (itN != nodeMap.end()) {
            const aiNode* p = itN->second->mParent;
            if (p) {
                auto itP = boneMap.find(p->mName.C_Str());
                if (itP != boneMap.end()) parentId = (int)itP->second;
            }
        }
        aiMatrix4x4 off;
        FindBoneOffset(scene, kv.first, off);
        unsortedBones.push_back({ kv.first, kv.second, parentId, off });
    }
    std::vector<TempBoneInfo> sortedBones;
    std::vector<int> newIndices(boneMap.size());
    std::vector<bool> added(boneMap.size(), false);
    int addedCount = 0;
    while ((size_t)addedCount < unsortedBones.size()) {
        for (const auto& bone : unsortedBones) {
            if (added[bone.originalIndex]) continue;
            if (bone.parentIndex == -1 || added[bone.parentIndex]) {
                newIndices[bone.originalIndex] = addedCount;
                sortedBones.push_back(bone);
                added[bone.originalIndex] = true;
                addedCount++;
            }
        }
    }
    json j;
    j["bones"] = json::array();
    for (size_t i = 0; i < sortedBones.size(); ++i) {
        auto& bone = sortedBones[i];
        finalBoneMap[bone.name] = static_cast<unsigned int>(i);
        if (bone.parentIndex != -1) {
            bone.parentIndex = newIndices[bone.parentIndex];
        }
        aiMatrix4x4 finalOffsetMatrix = bone.offsetMatrix;
        finalOffsetMatrix.a4 *= G_SCALE_FACTOR;
        finalOffsetMatrix.b4 *= G_SCALE_FACTOR;
        finalOffsetMatrix.c4 *= G_SCALE_FACTOR;
        json jb;
        jb["id"] = static_cast<unsigned int>(i);
        jb["name"] = bone.name;
        jb["parentId"] = bone.parentIndex;
        jb["offset"] = MatrixToJson(finalOffsetMatrix);
        j["bones"].push_back(jb);
    }
    std::ofstream out(outDir + "/skeleton.json");
    out << j.dump(2);
}

void processAnimation(unsigned idx, const aiAnimation* anim, const std::string& outDir) {
    json j;
    std::string name = anim->mName.C_Str();
    if (name.empty()) name = "anim_" + std::to_string(idx);
    j["name"] = name;
    j["duration"] = anim->mDuration;
    j["ticksPerSecond"] = (anim->mTicksPerSecond > 0.0 ? anim->mTicksPerSecond : 30.0);
    j["channels"] = json::array();
    for (unsigned c = 0; c < anim->mNumChannels; ++c) {
        const aiNodeAnim* ch = anim->mChannels[c];
        json jc;
        jc["bone"] = ch->mNodeName.C_Str();
        jc["posKeys"] = json::array();
        for (unsigned k = 0; k < ch->mNumPositionKeys; ++k) {
            const auto& pk = ch->mPositionKeys[k];
            jc["posKeys"].push_back({ {"t",pk.mTime}, {"x",pk.mValue.x * G_SCALE_FACTOR}, {"y",pk.mValue.y * G_SCALE_FACTOR}, {"z",pk.mValue.z * G_SCALE_FACTOR} });
        }
        jc["rotKeys"] = json::array();
        for (unsigned k = 0; k < ch->mNumRotationKeys; ++k) {
            const auto& rk = ch->mRotationKeys[k];
            jc["rotKeys"].push_back({ {"t",rk.mTime},{"x",rk.mValue.x},{"y",rk.mValue.y},{"z",rk.mValue.z},{"w",rk.mValue.w} });
        }
        jc["scaleKeys"] = json::array();
        for (unsigned k = 0; k < ch->mNumScalingKeys; ++k) {
            const auto& sk = ch->mScalingKeys[k];
            jc["scaleKeys"].push_back({ {"t",sk.mTime},{"x",sk.mValue.x},{"y",sk.mValue.y},{"z",sk.mValue.z} });
        }
        j["channels"].push_back(jc);
    }
    std::ofstream out(outDir + "/anim_" + std::to_string(idx) + ".anim");
    out << j.dump(2);
}

void processMaterial(unsigned int idx, const aiMaterial* mat, const aiScene* scene, const std::string& outDir)
{
    json j;
    aiColor4D diffuseColor;
    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor)) { j["diffuseColor"] = { diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a }; }
    else { j["diffuseColor"] = { 1.0f, 1.0f, 1.0f, 1.0f }; }
    aiString texturePath_ai;
    if (AI_SUCCESS == mat->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath_ai)) {
        std::string texturePath = texturePath_ai.C_Str();
        if (texturePath.rfind('*', 0) == 0) {
            int textureIndex = std::stoi(texturePath.substr(1));
            if (scene && textureIndex < (int)scene->mNumTextures) {
                const aiTexture* embeddedTexture = scene->mTextures[textureIndex];
                std::string extension = "png"; if (embeddedTexture->achFormatHint[0] != 0) { extension = embeddedTexture->achFormatHint; }
                std::string outputTextureFilename = "texture_" + std::to_string(idx) + "." + extension;
                if (embeddedTexture->mHeight == 0) {
                    std::ofstream textureFile(outDir + "/" + outputTextureFilename, std::ios::binary);
                    textureFile.write(reinterpret_cast<const char*>(embeddedTexture->pcData), embeddedTexture->mWidth);
                    textureFile.close();
                    j["diffuseTexture"] = outputTextureFilename;
                }
            }
        }
        else { std::filesystem::path p(texturePath); j["diffuseTexture"] = p.filename().string(); }
    }
    std::string path = outDir + "/material_" + std::to_string(idx) + ".material.json";
    std::ofstream out(path);
    out << j.dump(4);
}

void createSceneFile(const aiScene* scene, const std::string& outDir) {
    json j;
    j["mesh_count"] = scene->mNumMeshes;
    j["material_count"] = scene->mNumMaterials;
    j["animation_count"] = scene->mNumAnimations;
    j["meshes"] = json::array();
    for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
        json m;
        m["file"] = "mesh_" + std::to_string(i) + ".mesh";
        m["materialIndex"] = scene->mMeshes[i]->mMaterialIndex;
        j["meshes"].push_back(m);
    }
    j["materials"] = json::array();
    for (unsigned i = 0; i < scene->mNumMaterials; ++i) {
        j["materials"].push_back("material_" + std::to_string(i) + ".material.json");
    }
    j["animations"] = json::array();
    for (unsigned i = 0; i < scene->mNumAnimations; ++i) {
        j["animations"].push_back("anim_" + std::to_string(i) + ".anim");
    }
    j["skeleton"] = "skeleton.json";
    std::ofstream out(outDir + "/scene.json");
    out << j.dump(2);
}