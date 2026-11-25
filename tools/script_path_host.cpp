#include <cstddef>
#include <cstdint>

namespace {
// Rows from the schema-checked PSIN arrays supplied by the Python probe.
struct Node {
    std::uint32_t name;
    std::uint16_t parent, firstChild, nextSibling, previousSibling, firstAttribute;
    std::uint8_t attributeCount, flags;
};
struct Attribute {
    std::uint32_t name, textHash, value;
    std::uint8_t kind, flags;
    std::uint16_t reserved;
};
static_assert(sizeof(Node) == 16 && sizeof(Attribute) == 16);
constexpr unsigned noNode = 0xffff;

constexpr std::uint32_t nameHash(const char* s) {
    std::uint32_t h = 0;
    while (*s) {
        h += static_cast<unsigned char>(*s++);
        h += h << 10;
        h ^= h >> 6;
    }
    h += h << 3;
    h ^= h >> 11;
    return h + (h << 15);
}

int resolvePathImpl(
    const Node* nodes, unsigned nodeCount, const Attribute* attrs, unsigned attrCount,
    unsigned container, unsigned* active, bool relative, unsigned up,
    const std::uint32_t* path, const std::uint32_t* types, unsigned depth)
{
    if (!nodes || !attrs || !active || !path || !depth || nodeCount > noNode ||
        container >= nodeCount || *active >= nodeCount)
        return -3;
    unsigned parent = relative ? *active : container;
    for (unsigned i = 0; i < up; ++i) {
        if (parent == container) return -1;
        const unsigned next = nodes[parent].parent;
        if (next >= parent) return -3;
        parent = next;
    }
    for (unsigned level = 0; level < depth; ++level) {
        unsigned previous = noNode;
        unsigned match = noNode;
        bool ambiguous = false;
        unsigned visits = 0;
        for (unsigned i = nodes[parent].firstChild; i != noNode; i = nodes[i].nextSibling) {
            if (i >= nodeCount || ++visits > nodeCount) return -3;
            const auto& node = nodes[i];
            if (node.parent != parent || node.previousSibling != previous) return -3;
            previous = i;
            if (node.name != nameHash("state")) continue;
            const unsigned end = node.firstAttribute + node.attributeCount;
            if (node.attributeCount && end > attrCount) return -3;
            bool seenId = false;
            bool seenType = false;
            bool idMatch = false;
            bool typeMatch = (types == nullptr);
            for (unsigned a = node.firstAttribute; a < end; ++a) {
                if (attrs[a].name == nameHash("id")) {
                    if (seenId || attrs[a].kind != 5) return -3;
                    seenId = true;
                    idMatch = (attrs[a].textHash == path[level]);
                } else if (types && attrs[a].name == nameHash("type")) {
                    if (seenType || attrs[a].kind != 5) return -3;
                    seenType = true;
                    typeMatch = (attrs[a].textHash == types[level]);
                }
            }
            if (idMatch && typeMatch) {
                if (match != noNode)
                    ambiguous = true;
                else
                    match = i;
            }
        }
        if (ambiguous) return -2;
        if (match == noNode) return -1;
        parent = match;
    }
    *active = parent;
    return static_cast<int>(parent);
}
}

// Nonnegative results are source node indices. Negative results mean missing (-1),
// ambiguous (-2) or malformed (-3). The caller supplies the path to experiment with.
extern "C" __declspec(dllexport) int resolve_script_path(
    const Node* nodes, unsigned nodeCount, const Attribute* attrs, unsigned attrCount,
    unsigned container, unsigned* active, bool relative, unsigned up,
    const std::uint32_t* path, unsigned depth)
{
    return resolvePathImpl(nodes, nodeCount, attrs, attrCount, container, active, relative, up,
                           path, nullptr, depth);
}

// Type-qualified variant: each path segment must match both a label-kind `id` and a
// label-kind `type` attribute (`types[level]`) on the candidate `state` sibling.
extern "C" __declspec(dllexport) int resolve_script_typed_path(
    const Node* nodes, unsigned nodeCount, const Attribute* attrs, unsigned attrCount,
    unsigned container, unsigned* active, bool relative, unsigned up,
    const std::uint32_t* path, const std::uint32_t* types, unsigned depth)
{
    if (!types) return -3;
    return resolvePathImpl(nodes, nodeCount, attrs, attrCount, container, active, relative, up,
                           path, types, depth);
}
