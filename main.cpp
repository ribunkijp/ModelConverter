// ModelConverter.cpp — 导出 Mesh / Material / Skeleton / Anim / Scene

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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ---------------- 数据结构 ----------------

struct Vertex {
    float position[3]{};
    float texcoord[2]{};
    float normal[3]{};
    float tangent[3]{};
    int   boneIDs[4]{ -1, -1, -1, -1 };
    float weights[4]{ 0, 0, 0, 0 };
};

struct MeshHeader {
    uint32_t vertexCount{};
    uint32_t indexCount{};
    uint32_t materialIndex{};
};

// ---------------- 小工具 ----------------

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
    return json::array({ m.a1,m.a2,m.a3,m.a4, m.b1,m.b2,m.b3,m.b4, m.c1,m.c2,m.c3,m.c4, m.d1,m.d2,m.d3,m.d4 });
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

// ---------------- 函数声明 ----------------
void processMesh(unsigned, const aiMesh*, const std::string&, std::map<std::string, unsigned>&, unsigned&);
void processMaterial(unsigned, const aiMaterial*, const std::string&);
void processSkeleton(const aiScene*, const std::string&, const std::map<std::string, unsigned>&);
void processAnimation(unsigned, const aiAnimation*, const std::string&);
void createSceneFile(const aiScene*, const std::string&);

// ---------------- 主函数 ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "用法: ModelConverter.exe <输入文件.fbx>\n"; return 1; }

    std::filesystem::path inPath(argv[1]);
    if (!std::filesystem::exists(inPath)) { std::cerr << "错误: 文件不存在: " << inPath << "\n"; return 1; }

    std::filesystem::path abs = std::filesystem::absolute(inPath);
    std::string outDir = inPath.stem().string();
    std::filesystem::create_directories(outDir);

    auto logln = [](const std::string& s) { std::cout << s << std::endl; };

    logln("[Info] Input : " + abs.string());
    logln("[Info] Output: " + std::filesystem::absolute(outDir).string());


    Assimp::Importer importer;

#ifdef AI_CONFIG_IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS
    importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_READ_ALL_GEOMETRY_LAYERS, true);
#endif
#ifdef AI_CONFIG_IMPORT_FBX_OPTIMIZE_EMPTY_ANIMATION_CURVES
    importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_OPTIMIZE_EMPTY_ANIMATION_CURVES, false);
#endif
#ifdef AI_CONFIG_IMPORT_NO_SKELETON_MESHES
    importer.SetPropertyBool(AI_CONFIG_IMPORT_NO_SKELETON_MESHES, false);
#endif

    unsigned flags = aiProcess_Triangulate |
        aiProcess_ConvertToLeftHanded |
        aiProcess_GenNormals |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_OptimizeMeshes |
        aiProcess_SortByPType |
        aiProcess_CalcTangentSpace; // 这个标志确保了切线会被计算出来

    const aiScene* scene = importer.ReadFile(abs.u8string(), flags);
    if (!scene) { logln(std::string("[Error] Assimp: ") + importer.GetErrorString()); return 1; }

    {
        std::ostringstream oss;
        oss << "meshes=" << scene->mNumMeshes
            << " materials=" << scene->mNumMaterials
            << " animations=" << scene->mNumAnimations;
        logln(oss.str());
    }
    for (unsigned i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh* m = scene->mMeshes[i];
        std::ostringstream oss;
        oss << "  mesh[" << i << "] v=" << m->mNumVertices << " f=" << m->mNumFaces
            << " bones=" << m->mNumBones
            << " hasUV=" << (m->HasTextureCoords(0) ? "Y" : "N")
            << " hasN=" << (m->HasNormals() ? "Y" : "N");
        logln(oss.str());
    }

    std::map<std::string, unsigned> boneMap;
    unsigned boneCounter = 0;

    for (unsigned i = 0; i < scene->mNumMeshes; ++i)
        processMesh(i, scene->mMeshes[i], outDir, boneMap, boneCounter);
    { std::ostringstream oss; oss << "[Info] totalBones(from meshes) = " << boneMap.size(); logln(oss.str()); }

    for (unsigned i = 0; i < scene->mNumMaterials; ++i)
        processMaterial(i, scene->mMaterials[i], outDir);

    processSkeleton(scene, outDir, boneMap);

    for (unsigned i = 0; i < scene->mNumAnimations; ++i)
        processAnimation(i, scene->mAnimations[i], outDir);

    createSceneFile(scene, outDir);

    logln("[Info] Files in output dir:");
    for (auto& it : std::filesystem::directory_iterator(outDir))
        logln("  - " + it.path().filename().string());

    logln("模型已成功拆分到目录: " + outDir);
    return 0;
}

// ---------------- 实现：网格 ----------------
void processMesh(unsigned idx, const aiMesh* mesh, const std::string& outDir,
    std::map<std::string, unsigned>& boneMap, unsigned& boneCounter)
{
    std::vector<Vertex> vertices(mesh->mNumVertices);
    for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
        vertices[i].position[0] = mesh->mVertices[i].x;
        vertices[i].position[1] = mesh->mVertices[i].y;
        vertices[i].position[2] = mesh->mVertices[i].z;
       
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
        std::string name = b->mName.C_Str();
        unsigned id;
        auto it = boneMap.find(name);
        if (it == boneMap.end()) { id = boneCounter++; boneMap[name] = id; }
        else id = it->second;
        for (unsigned wi = 0; wi < b->mNumWeights; ++wi) {
            unsigned vId = b->mWeights[wi].mVertexId; float w = b->mWeights[wi].mWeight;
            if (vId < vertices.size() && w > 0.0f) AddBoneWeight(vertices[vId], (int)id, w);
        }
    }
    for (auto& v : vertices) NormalizeWeights(v);

    std::vector<uint32_t> indices; indices.reserve(mesh->mNumFaces * 3);
    for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
        const aiFace& face = mesh->mFaces[f];
        for (unsigned j = 0; j < face.mNumIndices; ++j) indices.push_back(face.mIndices[j]);
    }

    std::string path = outDir + "/mesh_" + std::to_string(idx) + ".mesh";
    std::ofstream out(path, std::ios::binary);
    MeshHeader header{ (uint32_t)vertices.size(), (uint32_t)indices.size(), mesh->mMaterialIndex };
    out.write((char*)&header, sizeof(header));
    out.write((char*)vertices.data(), vertices.size() * sizeof(Vertex));
    out.write((char*)indices.data(), indices.size() * sizeof(uint32_t));
}

// ---------------- 实现：材质 ----------------
void processMaterial(unsigned idx, const aiMaterial* mat, const std::string& outDir) {
    json j;

    aiColor4D diffuseColor;
    if (AI_SUCCESS == aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuseColor)) {
        j["diffuseColor"] = { diffuseColor.r, diffuseColor.g, diffuseColor.b, diffuseColor.a };
    }
    else {
        j["diffuseColor"] = { 1.0f, 1.0f, 1.0f, 1.0f };
    }

    aiString texturePath;
    if (AI_SUCCESS == mat->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath)) {
        std::filesystem::path p(texturePath.C_Str());
        j["diffuseTexture"] = p.filename().string();
    }

    std::string path = outDir + "/material_" + std::to_string(idx) + ".material.json";
    std::ofstream out(path);
    out << j.dump(4);
}

// ---------------- 实现：骨架 ----------------
void processSkeleton(const aiScene* scene, const std::string& outDir,
    const std::map<std::string, unsigned>& boneMap)
{
    std::unordered_map<std::string, const aiNode*> nodeMap; BuildNodeMap(scene->mRootNode, nodeMap);

    json j; j["bones"] = json::array();
    for (const auto& kv : boneMap) {
        const std::string& name = kv.first; unsigned id = kv.second;
        int parentId = -1;
        auto itN = nodeMap.find(name);
        if (itN != nodeMap.end()) {
            const aiNode* p = itN->second->mParent;
            if (p) { auto itP = boneMap.find(p->mName.C_Str()); if (itP != boneMap.end()) parentId = (int)itP->second; }
        }
        aiMatrix4x4 off;
        if (!FindBoneOffset(scene, name, off)) off = aiMatrix4x4();
        json jb; jb["id"] = id; jb["name"] = name; jb["parentId"] = parentId; jb["offset"] = MatrixToJson(off);
        j["bones"].push_back(jb);
    }
    std::ofstream out(outDir + "/skeleton.json"); out << j.dump(2);
}

// ---------------- 实现：动画 ----------------
void processAnimation(unsigned idx, const aiAnimation* anim, const std::string& outDir) {
    json j;
    std::string name = anim->mName.C_Str(); if (name.empty()) name = "anim_" + std::to_string(idx);
    j["name"] = name;
    j["duration"] = anim->mDuration;
    j["ticksPerSecond"] = (anim->mTicksPerSecond > 0.0 ? anim->mTicksPerSecond : 30.0);
    j["channels"] = json::array();

    for (unsigned c = 0; c < anim->mNumChannels; ++c) {
        const aiNodeAnim* ch = anim->mChannels[c];
        json jc; jc["bone"] = ch->mNodeName.C_Str();
        jc["posKeys"] = json::array();
        for (unsigned k = 0; k < ch->mNumPositionKeys; ++k) {
            const auto& pk = ch->mPositionKeys[k];
            jc["posKeys"].push_back({ {"t",pk.mTime},{"x",pk.mValue.x},{"y",pk.mValue.y},{"z",pk.mValue.z} });
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

// ---------------- 实现：场景 ----------------
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