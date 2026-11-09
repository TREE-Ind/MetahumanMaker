// Create MetaHuman's single-frame Unreal groom Alembic writer.
// Uses the open-source Alembic and Imath libraries; see bundled notices.
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace A = Alembic::Abc;
namespace G = Alembic::AbcGeom;

template<class T> std::vector<T> read_array(std::ifstream& input, size_t count) {
    std::vector<T> values(count);
    if (!input.read(reinterpret_cast<char*>(values.data()), sizeof(T) * count))
        throw std::runtime_error("Incomplete groom buffer");
    return values;
}

template<class T> void finite(const std::vector<T>& values, size_t components) {
    const float* data = reinterpret_cast<const float*>(values.data());
    for (size_t i = 0; i < values.size() * components; ++i)
        if (!std::isfinite(data[i])) throw std::runtime_error("Nonfinite groom data");
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Expected input buffer and output Alembic paths");
        std::ifstream input(std::filesystem::path(argv[1]), std::ios::binary);
        char magic[8];
        if (!input.read(magic, 8) || std::string(magic, 8) != "CMGABC01")
            throw std::runtime_error("Invalid groom buffer version");
        auto header = read_array<uint32_t>(input, 3);
        const uint32_t strands = header[0], points = header[1], colors_present = header[2];
        if (!strands || strands > 10000000 || points < strands * uint64_t(2) ||
            points > 100000000 || colors_present > 1)
            throw std::runtime_error("Invalid groom buffer dimensions");
        auto counts = read_array<int32_t>(input, strands);
        uint64_t total = 0;
        for (auto count : counts) {
            if (count < 2) throw std::runtime_error("Each strand needs two points");
            total += count;
        }
        if (total != points) throw std::runtime_error("Groom point count mismatch");
        static_assert(sizeof(Imath::V3f) == 12 && sizeof(Imath::V2f) == 8);
        auto positions = read_array<Imath::V3f>(input, points);
        auto widths = read_array<float>(input, points);
        auto groups = read_array<int32_t>(input, strands);
        auto guides = read_array<int32_t>(input, strands);
        auto uv = read_array<Imath::V2f>(input, strands);
        std::vector<Imath::C3f> colors;
        if (colors_present) colors = read_array<Imath::C3f>(input, points);
        if (input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("Unexpected trailing groom data");
        finite(positions, 3); finite(widths, 1); finite(uv, 2); finite(colors, 3);
        for (auto width : widths)
            if (width < 0) throw std::runtime_error("Negative strand width");
        for (uint32_t i = 0; i < strands; ++i)
            if (groups[i] < 0 || (guides[i] != 0 && guides[i] != 1))
                throw std::runtime_error("Invalid groom group or guide flag");
        std::ofstream output(std::filesystem::path(argv[2]), std::ios::binary);
        if (!output) throw std::runtime_error("Cannot create output Alembic");
        {
            A::OArchive archive(Alembic::AbcCoreOgawa::WriteArchive()(&output,
                                Alembic::AbcCoreAbstract::MetaData()));
            G::OCurves curves(archive.getTop(), "Groom");
            auto& schema = curves.getSchema();
            G::OCurvesSchema::Sample sample;
            sample.setPositions(A::P3fArraySample(positions));
            sample.setCurvesNumVertices(A::Int32ArraySample(counts));
            sample.setType(G::kLinear);
            sample.setWrap(G::kNonPeriodic);
            sample.setWidths(G::OFloatGeomParam::Sample(A::FloatArraySample(widths), G::kVertexScope));
            schema.set(sample);
            auto params = schema.getArbGeomParams();
            G::OInt32GeomParam group(params, "groom_group_id", false, G::kUniformScope, 1);
            group.set(G::OInt32GeomParam::Sample(A::Int32ArraySample(groups), G::kUniformScope));
            G::OInt32GeomParam guide(params, "groom_guide", false, G::kUniformScope, 1);
            guide.set(G::OInt32GeomParam::Sample(A::Int32ArraySample(guides), G::kUniformScope));
            G::OV2fGeomParam root_uv(params, "groom_root_uv", false, G::kUniformScope, 1);
            root_uv.set(G::OV2fGeomParam::Sample(A::V2fArraySample(uv), G::kUniformScope));
            G::OFloatGeomParam width(params, "groom_width", false, G::kVertexScope, 1);
            width.set(G::OFloatGeomParam::Sample(A::FloatArraySample(widths), G::kVertexScope));
            if (colors_present) {
                G::OC3fGeomParam color(params, "groom_color", false, G::kVertexScope, 1);
                color.set(G::OC3fGeomParam::Sample(A::C3fArraySample(colors), G::kVertexScope));
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("Writing output Alembic failed");
        std::cout << "Groom Alembic written: " << strands << " strands, " << points << " points\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
