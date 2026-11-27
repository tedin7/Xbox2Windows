// MTBUF uses the instruction's explicit format and the descriptor's byte range.
// Format numbers follow tools/disasm_rdna2.py MTBUF_FORMATS. Conversion and
// unaligned addressing follow gfx2hlsl.tbuffer / ldu, cross-checked against
// Victory 9376fd6 format_load. Only the runtime-checked identity raw SRV is used, or
// its byte window (the byte-address rewrite of the descriptor, read as the same raw view).
enum class ChannelNumber { unorm, snorm, uscaled, sscaled, uint_, sint, float_ };
struct MtbufFamily {
    unsigned first, last, bits, components;
    ChannelNumber first_number;
};
string normalized_channel(const string &field, unsigned width, bool snorm) {
    // p/(2^k-1) has binary expansion 0.ppp... (p padded to k bits).
    // Rotate the first set bit to the top, repeat at least 25 bits, then
    // round a 24-bit significand. A nonzero repeating tail is never a tie.
    const unsigned k = width - unsigned(snorm);
    const auto max = to_string((1u << k) - 1);
    string p = field, sign = "0";
    if (snorm) {
        sign = ir.binary("and", field, "2147483648");
        p = select(cmp("slt", field, "0"), ir.binary("sub", "0", field), field);
        p = select(cmp("ugt", p, max), max, p);
    }
    // p has at most 16 bits; converting it to binary32 is exact.
    auto exp = ir.binary("lshr", ir.value("bitcast float " + ir.value("uitofp i32 " + p + " to float") + " to i32"), "23");
    auto distance = ir.binary("and", ir.binary("sub", to_string(126 + k), exp), "31");
    auto right = ir.binary("and", ir.binary("sub", to_string(k), distance), "31");
    auto period = ir.binary("and", ir.binary("or", ir.binary("shl", p, distance), ir.binary("lshr", p, right)), max);
    unsigned periods = (25 + k - 1) / k, precision = periods * k;
    unsigned repeats = 0;
    for (unsigned i = 0; i < periods; ++i)
        repeats |= 1u << (i * k);
    auto digits = ir.binary("mul", period, to_string(repeats));
    auto rounded = ir.binary("add", ir.binary("lshr", digits, to_string(precision - 24)),
                             ir.binary("and", ir.binary("lshr", digits, to_string(precision - 25)), "1"));
    auto base = ir.binary("shl", ir.binary("sub", exp, to_string(k + 1)), "23");
    auto magnitude = select(cmp("eq", p, "0"), "0", ir.binary("add", base, rounded));
    return snorm ? ir.binary("or", magnitude, sign) : magnitude;
}
void formatted_byte_load(const Inst &i, const State &s) {
    constexpr MtbufFamily families[] = {
        {1, 6, 8, 1, ChannelNumber::unorm}, {7, 13, 16, 1, ChannelNumber::unorm},
        {14, 19, 8, 2, ChannelNumber::unorm}, {20, 22, 32, 1, ChannelNumber::uint_},
        {23, 29, 16, 2, ChannelNumber::unorm}, {56, 61, 8, 4, ChannelNumber::unorm},
        {62, 64, 32, 2, ChannelNumber::uint_}, {65, 71, 16, 4, ChannelNumber::unorm},
        {72, 74, 32, 3, ChannelNumber::uint_}, {75, 77, 32, 4, ChannelNumber::uint_}
    };
    const unsigned format = (i.a >> 19) & 127, count = i.op + 1;
    const auto family = std::find_if(std::begin(families), std::end(families), [&](const auto &f) {
        return format >= f.first && format <= f.last;
    });
    if (family == std::end(families) || count > family->components)
        refuse(i.pc, "unsupported MTBUF format or absent component");
    if (i.dst + count > 256)
        refuse(i.pc, "MTBUF destination tuple overflow");
    const unsigned width = family->bits;
    const auto numeric = static_cast<ChannelNumber>(unsigned(family->first_number) + format - family->first);
    // The byte-address rewrite of a descriptor (stride cleared in word 1) is read as the same raw view, as for MUBUF.
    const unsigned base = (i.b >> 16 & 31) * 4;
    const auto handle = resource(i, s, base, s.at(base + 1).type == Symbol::Type::zero_stride_word
                                                 ? ComputeViewKind::byte_window_srv
                                                 : ComputeViewKind::raw_vector_srv, 4);
    const unsigned scalar_offset = i.b >> 24;
    auto addr = to_string(i.a & 4095);
    if (scalar_offset != 125 && scalar_offset != 128)
        addr = ir.binary("add", addr, operand32(i, scalar_offset));
    if (i.a & 0x1000)
        addr = ir.binary("add", addr, read(256 + (i.b & 255)));
    // RDNA2 checks the whole formatted element. Desktop raw loads by
    // themselves can return some in-range components from a truncated element.
    auto dimensions = ir.invoke("%dx.types.Dimensions", "getDimensions", 72,
                                {{"%dx.types.Handle", handle}, {"i32", "undef"}}, "nounwind readonly");
    auto bytes = ir.value("extractvalue %dx.types.Dimensions " + dimensions + ", 0");
    auto size = to_string(width * family->components / 8);
    auto valid = ir.value("and i1 " + cmp("uge", bytes, size) + ", " +
                          cmp("ule", addr, ir.binary("sub", bytes, size)));
    const auto aligned = ir.binary("and", addr, "4294967292");
    const auto shift_bits = ir.binary("shl", ir.binary("and", addr, "3"), "3");
    const unsigned word_count = (count * width + 31) / 32;
    const auto chunk = load_buffer(handle, aligned, true, word_count);
    std::array<string, 5> words;
    for (unsigned j = 0; j < word_count; ++j)
        words[j] = ir.value("extractvalue %dx.types.ResRet.i32 " + chunk + ", " + to_string(j));
    if (count * width > 8) {
        const auto next = ir.binary("add", aligned, to_string(word_count * 4));
        const auto high = load_buffer(handle, next, true, 1);
        words[word_count] = select(cmp("ult", next, aligned), "0",
                                   ir.value("extractvalue %dx.types.ResRet.i32 " + high + ", 0"));
    }
    // Snapshot all address-derived values before writing destinations: OFFEN's
    // VGPR can be any member of the output tuple.
    const auto aligned_access = cmp("eq", shift_bits, "0");
    const auto inverse = ir.binary("and", ir.binary("sub", "0", shift_bits), "31");
    for (unsigned j = 0; j < word_count; ++j) {
        const auto low = ir.binary("lshr", words[j], shift_bits);
        words[j] = words[j + 1].empty() ? low :
            ir.binary("or", low, select(aligned_access, "0", ir.binary("shl", words[j + 1], inverse)));
    }
    for (unsigned j = 0; j < count; ++j) {
        auto field = words[j * width / 32];
        if (width < 32) {
            field = ir.binary("lshr", field, to_string(j * width % 32));
            field = ir.binary("and", field, to_string((1u << width) - 1));
        }
        const bool signed_field = numeric == ChannelNumber::sint || numeric == ChannelNumber::snorm ||
                                  numeric == ChannelNumber::sscaled;
        if (signed_field && width < 32) {
            auto distance = to_string(32 - width);
            field = ir.binary("ashr", ir.binary("shl", field, distance), distance);
        }
        if (numeric == ChannelNumber::float_ && width == 16) {
            auto f = ir.invoke("float", "legacyF16ToF32", 131, {{"i32", field}}, "nounwind readnone");
            field = ir.value("bitcast float " + f + " to i32");
        } else if (numeric == ChannelNumber::unorm || numeric == ChannelNumber::snorm) {
            field = normalized_channel(field, width, signed_field);
        } else if (numeric == ChannelNumber::uscaled || numeric == ChannelNumber::sscaled) {
            auto f = ir.value(string(signed_field ? "sitofp" : "uitofp") + " i32 " + field + " to float");
            field = ir.value("bitcast float " + f + " to i32");
        }
        vector_write(256 + i.dst + j, select(valid, field, "0"));
    }
}
