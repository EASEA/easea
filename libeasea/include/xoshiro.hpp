/* Code was taken from here:
   https://prng.di.unimi.it
   And wrapped as a bit generator for C++'s random module' */

/*  Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* Code was modified by Léo Chéneau to remove some warnings and keep only Xoshiro256PP
 * Original repository: https://github.com/david-cortes/xoshiro_cpp */

#include <iostream>
#include <cstring>
#include <cstdint>

namespace xoshiro {

#if __cplusplus >= 202001L
#   define HAS_CPP20
#endif

static inline std::uint64_t rotl64(const std::uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

constexpr static const uint64_t JUMP_X256PP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

constexpr static const uint64_t LONG_JUMP_X256PP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

template <class int_t, class rng_t>
static inline void jump_state(const int_t jump_table[4], rng_t &rng)
{
    int_t s0 = 0;
    int_t s1 = 0;
    int_t s2 = 0;
    int_t s3 = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int b = 0; b < static_cast<int>(8*sizeof(int_t)); b++)
        {
            if (jump_table[i] & ((int_t)1) << b)
            {
                s0 ^= rng.state[0];
                s1 ^= rng.state[1];
                s2 ^= rng.state[2];
                s3 ^= rng.state[3];
            }
            rng();
        }
    }

    rng.state[0] = s0;
    rng.state[1] = s1;
    rng.state[2] = s2;
    rng.state[3] = s3;
}

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and 
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */
static inline std::uint64_t splitmix64(const std::uint64_t seed)
{
    std::uint64_t z = (seed + 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/* This is xoshiro256++ 1.0, one of our all-purpose, rock-solid generators.
   It has excellent (sub-ns) speed, a state (256 bits) that is large
   enough for any parallel application, and it passes all tests we are
   aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */
class Xoshiro256PP
{
public:
    using result_type = std::uint64_t;
    std::uint64_t state[4] = {0x3d23dce41c588f8c, 0x10c770bb8da027b0, 0xc7a4c5e87c63ba25, 0xa830f83239465a2e};

    constexpr static result_type min()
    {
        return 0;
    }

    constexpr static result_type max()
    {
        return UINT64_MAX;
    }

    Xoshiro256PP() = default;

    ~Xoshiro256PP() noexcept = default;

    Xoshiro256PP(Xoshiro256PP &other) = default;

    Xoshiro256PP& operator=(const Xoshiro256PP &other) = default;

    Xoshiro256PP(Xoshiro256PP &&) noexcept = default;

    Xoshiro256PP& operator=(Xoshiro256PP &&) noexcept = default;

    void seed(const std::uint64_t seed)
    {
        this->state[0] = splitmix64(splitmix64(seed));
        this->state[1] = splitmix64(this->state[0]);
        this->state[2] = splitmix64(this->state[1]);
        this->state[3] = splitmix64(this->state[2]);
    }

    void seed(const std::uint64_t seed[4])
    {
        std::memcpy(this->state, seed, 4*sizeof(std::uint64_t));
    }

    template<class Sseq>
    void seed(Sseq& seq)
    {
        seq.generate(reinterpret_cast<std::uint32_t*>(&this->state[0]),
                     reinterpret_cast<std::uint32_t*>(&this->state[0] + 4));
    }

    Xoshiro256PP(const std::uint64_t seed)
    {
        this->seed(seed);
    }

    Xoshiro256PP(const std::uint64_t seed[4])
    {
        this->seed(seed);
    }

    template<class Sseq>
    Xoshiro256PP(Sseq& seq)
    {
        this->seed(seq);
    }

    result_type operator()()
    {
        const std::uint64_t result = rotl64(this->state[0] + this->state[3], 23) + this->state[0];
        const std::uint64_t t = this->state[1] << 17;
        this->state[2] ^= this->state[0];
        this->state[3] ^= this->state[1];
        this->state[1] ^= this->state[2];
        this->state[0] ^= this->state[3];
        this->state[2] ^= t;
        this->state[3] = rotl64(this->state[3], 45);
        return result;
    }

    void discard(unsigned long long z)
    {
        for (unsigned long long ix = 0; ix < z; ix++)
            this->operator()();
    }

    Xoshiro256PP jump()
    {
        Xoshiro256PP new_gen = *this;
        jump_state(JUMP_X256PP, new_gen);
        return new_gen;
    }

    Xoshiro256PP long_jump()
    {
        Xoshiro256PP new_gen = *this;
        jump_state(LONG_JUMP_X256PP, new_gen);
        return new_gen;
    }

    bool operator==(const Xoshiro256PP &rhs)
    {
        return std::memcmp(this->state, rhs.state, 4*sizeof(std::uint64_t));
    }

    #ifndef HAS_CPP20
    bool operator!=(const Xoshiro256PP &rhs)
    {
        return !std::memcmp(this->state, rhs.state, 4*sizeof(std::uint64_t));
    }
    #endif

    template< class CharT, class Traits >
    friend std::basic_ostream<CharT,Traits>&
    operator<<(std::basic_ostream<CharT,Traits>& ost, const Xoshiro256PP& e)
    {
        ost.write(reinterpret_cast<const char*>(&e.state[0]), sizeof(std::uint64_t));
        ost.put(' ');
        ost.write(reinterpret_cast<const char*>(&e.state[1]), sizeof(std::uint64_t));
        ost.put(' ');
        ost.write(reinterpret_cast<const char*>(&e.state[2]), sizeof(std::uint64_t));
        ost.put(' ');
        ost.write(reinterpret_cast<const char*>(&e.state[3]), sizeof(std::uint64_t));
        return ost;
    }
    template< class CharT, class Traits >
    friend std::basic_istream<CharT,Traits>&
    operator>>(std::basic_istream<CharT,Traits>& ist, Xoshiro256PP& e)
    {
        ist.read(reinterpret_cast<char*>(&e.state[0]), sizeof(std::uint64_t));
        ist.get();
        ist.read(reinterpret_cast<char*>(&e.state[1]), sizeof(std::uint64_t));
        ist.get();
        ist.read(reinterpret_cast<char*>(&e.state[2]), sizeof(std::uint64_t));
        ist.get();
        ist.read(reinterpret_cast<char*>(&e.state[3]), sizeof(std::uint64_t));
        return ist;
    }
};

}
