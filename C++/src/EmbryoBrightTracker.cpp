// EmbryoBrightTracker.cpp
#include "EmbryoBrightTracker.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static void drawDashedLine(
    cv::Mat& img,
    cv::Point p0,
    cv::Point p1,
    const cv::Scalar& color,
    int thickness,
    int dashLen = 8,
    int gapLen  = 6)
{
    cv::LineIterator it(img, p0, p1, 8);
    int on = dashLen;
    int off = gapLen;
    bool draw = true;
    int cnt = 0;

    for (int i = 0; i < it.count; ++i, ++it) {
        if (draw) {
            cv::circle(img, it.pos(), thickness / 2, color, -1);
        }
        cnt++;
        if (draw && cnt >= on) { draw = false; cnt = 0; }
        else if (!draw && cnt >= off) { draw = true; cnt = 0; }
    }
}

static void drawDashedRect(
    cv::Mat& img,
    const cv::Rect& rc,
    const cv::Scalar& color,
    int thickness,
    int dashLen = 8,
    int gapLen  = 6)
{
    cv::Point p1(rc.x, rc.y);
    cv::Point p2(rc.x + rc.width, rc.y);
    cv::Point p3(rc.x + rc.width, rc.y + rc.height);
    cv::Point p4(rc.x, rc.y + rc.height);

    drawDashedLine(img, p1, p2, color, thickness, dashLen, gapLen);
    drawDashedLine(img, p2, p3, color, thickness, dashLen, gapLen);
    drawDashedLine(img, p3, p4, color, thickness, dashLen, gapLen);
    drawDashedLine(img, p4, p1, color, thickness, dashLen, gapLen);
}

EmbryoBrightTracker::EmbryoBrightTracker(const BaseConfig &cfg, const std::string &outputDir)
    : config(cfg), outDir(outputDir) {
}

std::vector<cv::Mat> EmbryoBrightTracker::loadVolume(const fs::path& imageFile)
{
    std::vector<cv::Mat> pages;
    // ************************* read directly by openCV
    if (!cv::imreadmulti(imageFile.string(), pages, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE)) {
        throw std::runtime_error("imreadmulti failed: " + imageFile.string());
    }

    // --- Pass 1: convert all pages to float, compute GLOBAL min/max over whole volume ---
    std::vector<cv::Mat> tmp;
    tmp.reserve(pages.size());

    double gmn = 0.0, gmx = 0.0;
    bool first = true;

    for (auto& p : pages) {
        cv::Mat f;
        p.convertTo(f, CV_32F);
        tmp.push_back(f);

        double mn = 0.0, mx = 0.0;
        cv::minMaxLoc(f, &mn, &mx);
        if (first) { gmn = mn; gmx = mx; first = false; }
        else {
            gmn = std::min(gmn, mn);
            gmx = std::max(gmx, mx);
        }
    }

    // --- Pass 2: normalize every slice with SAME (gmn,gmx) ---
    std::cout << "[DBG][loadVolume] file=" << imageFile.filename().string()
          << " Z=" << pages.size()
          << " gmn=" << gmn << " gmx=" << gmx
          << " (gmx-gmn)=" << (gmx - gmn)
          << "\n";
    std::vector<cv::Mat> vol;
    vol.reserve(tmp.size());

    if (gmx > gmn) {
        float denom = (float)(gmx - gmn);
        float fmn = (float)gmn;
        for (auto& f : tmp) {
            cv::Mat n = (f - fmn) / denom;
            vol.push_back(n);
        }
    } else {
        // degenerate volume: all same intensity
        for (size_t i = 0; i < tmp.size(); ++i) {
            vol.push_back(cv::Mat::zeros(tmp[0].rows, tmp[0].cols, CV_32F));
        }
    }

    return vol;

}

cv::Mat EmbryoBrightTracker::zSliceToU8(const cv::Mat &zf) const {
    cv::Mat u8;
    cv::Mat tmp = zf;
    if (tmp.type() != CV_32F) {
        tmp.convertTo(tmp, CV_32F);
    }
    cv::Mat clipped;
    cv::min(tmp, 1.0f, clipped);
    cv::max(clipped, 0.0f, clipped);
    clipped.convertTo(u8, CV_8U, 255.0);
    return u8;
}

float EmbryoBrightTracker::percentileThreshold(const std::vector<cv::Mat> &vol, float pct) const {
    std::vector<float> samples;
    samples.reserve(200000);

    int Z = (int) vol.size();
    if (Z == 0) return 0.5f;

    for (int z = 0; z < Z; ++z) {
        const cv::Mat &m = vol[z];
        int step = std::max(1, (m.rows * m.cols) / 20000);
        for (int i = 0; i < m.rows; ++i) {
            const float *row = m.ptr<float>(i);
            for (int j = 0; j < m.cols; j += step) {
                samples.push_back(row[j]);
            }
        }
    }

    if (samples.empty()) return 0.5f;
    std::sort(samples.begin(), samples.end());
    int idx = (int) std::round((pct / 100.0f) * (samples.size() - 1));
    idx = std::max(0, std::min((int) samples.size() - 1, idx));
    return samples[idx];
}

EmbryoBrightTracker::BBox3D EmbryoBrightTracker::makeBBox(

    const cv::Point3f &c, float matureDiam, int Z, int Y, int X) const {

    // Anisotropic bbox:
    // XY in pixel units, Z is much thinner (original data has only ~35 slices).
    // If we use same radius for Z, bbox will clamp to [0, Z-1] => "flattened" synth.
    float rXY = 0.45f * matureDiam;      // was 0.90f (too big)
    rXY = std::max(rXY, 18.0f);          // safety floor

    // z radius: smaller than xy (empirical z anisotropy ~7 from your config, but we hardcode to avoid dependency)
    const float Z_SCALE = 7.0f;
    float rZ = rXY / Z_SCALE;
    rZ = std::max(rZ, 2.0f);             // at least 2 slices
    rZ = std::min(rZ, (float)std::max(3, Z / 3));  // prevent full-stack coverage

    int z0 = (int)std::floor(c.z - rZ);
    int z1 = (int)std::ceil (c.z + rZ);
    int y0 = (int)std::floor(c.y - rXY);
    int y1 = (int)std::ceil (c.y + rXY);
    int x0 = (int)std::floor(c.x - rXY);
    int x1 = (int)std::ceil (c.x + rXY);

    z0 = std::max(0, z0);
    z1 = std::min(Z - 1, z1);
    y0 = std::max(0, y0);
    y1 = std::min(Y - 1, y1);
    x0 = std::max(0, x0);
    x1 = std::min(X - 1, x1);

    return {z0, z1, y0, y1, x0, x1};
}

static float estimateDiameterFromVoxelCount(int voxelCount) {
    if (voxelCount <= 0) return 0.0f;
    float V = (float) voxelCount;
    float r = std::cbrt((3.0f * V) / (4.0f * (float) M_PI));
    return 2.0f * r;
}

struct UF {
    std::vector<int> p, r;
    UF(int n = 0) { init(n); }

    void init(int n) {
        p.resize(n);
        r.assign(n, 0);
        for (int i = 0; i < n; ++i) p[i] = i;
    }

    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
    }
};

struct Peak3D {
    int z, y, x;
    float v;
};

static inline float dist2_3d(const cv::Point3f &a, const cv::Point3f &b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// redudant
static bool isLocalMax3x3x3(const std::vector<cv::Mat> &vol, int z, int y, int x, float v) {
    int Z = (int) vol.size();
    int Y = vol[0].rows;
    int X = vol[0].cols;

    for (int dz = -1; dz <= 1; ++dz) {
        int zz = z + dz;
        if (zz < 0 || zz >= Z) continue;
        const cv::Mat &m = vol[zz];
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= Y) continue;
            const float *row = m.ptr<float>(yy);
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= X) continue;
                if (dz == 0 && dy == 0 && dx == 0) continue;
                if (row[xx] > v) return false; // strict greater: keep only true peaks
            }
        }
    }
    return true;
}

static bool refinePeakWeightedCentroid(
    const std::vector<cv::Mat> &vol,
    const cv::Point3f &seed,
    float thresh2,
    int refineR,
    cv::Point3f &outCenter,
    int &outVoxCnt,
    float &outMeanI) {
    int Z = (int) vol.size();
    int Y = vol[0].rows;
    int X = vol[0].cols;

    int z0 = std::max(0, (int) std::floor(seed.z - refineR));
    int z1 = std::min(Z - 1, (int) std::ceil(seed.z + refineR));
    int y0 = std::max(0, (int) std::floor(seed.y - refineR));
    int y1 = std::min(Y - 1, (int) std::ceil(seed.y + refineR));
    int x0 = std::max(0, (int) std::floor(seed.x - refineR));
    int x1 = std::min(X - 1, (int) std::ceil(seed.x + refineR));

    double sumW = 0, sx = 0, sy = 0, sz = 0, sumI = 0;
    int vox = 0;

    for (int z = z0; z <= z1; ++z) {
        const cv::Mat &m = vol[z];
        for (int y = y0; y <= y1; ++y) {
            const float *row = m.ptr<float>(y);
            for (int x = x0; x <= x1; ++x) {
                float v = row[x];
                if (v < thresh2) continue;

                // optional: spherical window (stronger suppression of nearby noise)
                float dz = (float) z - seed.z;
                float dy = (float) y - seed.y;
                float dx = (float) x - seed.x;
                if (dx * dx + dy * dy + dz * dz > (float) (refineR * refineR)) continue;

                double w = (double) v;
                sumW += w;
                sx += w * x;
                sy += w * y;
                sz += w * z;
                sumI += v;
                vox++;
            }
        }
    }

    if (vox <= 0 || sumW <= 1e-9) return false;

    outCenter = cv::Point3f((float) (sx / sumW), (float) (sy / sumW), (float) (sz / sumW));
    outVoxCnt = vox;
    outMeanI = (float) (sumI / (double) vox);
    return true;
}

std::vector<EmbryoBrightTracker::Comp3DStat> EmbryoBrightTracker::extractConnectedComponents3D(
    const std::vector<cv::Mat> &vol,
    float threshLow,
    int z0, int z1, int y0, int y1, int x0, int x1,
    bool use26) const {
    int Z = (int) vol.size();
    int Y = vol[0].rows;
    int X = vol[0].cols;

    z0 = std::max(0, z0);
    z1 = std::min(Z - 1, z1);
    y0 = std::max(0, y0);
    y1 = std::min(Y - 1, y1);
    x0 = std::max(0, x0);
    x1 = std::min(X - 1, x1);

    int rz = z1 - z0 + 1;
    int ry = y1 - y0 + 1;
    int rx = x1 - x0 + 1;
    if (rz <= 0 || ry <= 0 || rx <= 0) return {};

    auto lin = [=](int z, int y, int x) {
        return ((z - z0) * ry + (y - y0)) * rx + (x - x0);
    };

    std::vector<uint8_t> mask((size_t) rz * ry * rx, 0);

    size_t maskOn = 0;
    size_t maskTot = (size_t)rz * ry * rx;

    std::vector<uint8_t> vis((size_t) rz * ry * rx, 0);

    // Build mask
    for (int z = z0; z <= z1; ++z) {
        const cv::Mat &m = vol[z];
        for (int y = y0; y <= y1; ++y) {
            const float *row = m.ptr<float>(y);
            for (int x = x0; x <= x1; ++x) {
                if (row[x] >= threshLow) { mask[(size_t)lin(z,y,x)] = 1; maskOn++; }
            }
        }
    }

    // Neighbor offsets
    std::vector<cv::Point3i> nbr;
    nbr.reserve(use26 ? 26 : 6);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dz == 0 && dy == 0 && dx == 0) continue;
                if (!use26) {
                    int man = std::abs(dx) + std::abs(dy) + std::abs(dz);
                    if (man != 1) continue;
                }
                nbr.push_back(cv::Point3i(dx, dy, dz));
            }
        }
    }

    std::vector<Comp3DStat> comps;
    comps.reserve(64);

    std::vector<cv::Point3i> stack;
    stack.reserve(200000);

    for (int z = z0; z <= z1; ++z) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                int id = lin(z, y, x);
                if (!mask[(size_t) id] || vis[(size_t) id]) continue;

                Comp3DStat st;
                st.z0 = st.z1 = z;
                st.y0 = st.y1 = y;
                st.x0 = st.x1 = x;

                // BFS/DFS
                vis[(size_t) id] = 1;
                stack.clear();
                stack.push_back(cv::Point3i(x, y, z));

                while (!stack.empty()) {
                    cv::Point3i p = stack.back();
                    stack.pop_back();

                    float v = vol[p.z].ptr<float>(p.y)[p.x];
                    double w = (double) v;

                    st.vox++;
                    st.sumW += w;
                    st.sx += w * p.x;
                    st.sy += w * p.y;
                    st.sz += w * p.z;
                    st.ux += p.x;
                    st.uy += p.y;
                    st.uz += p.z;
                    st.sumI += v;

                    st.z0 = std::min(st.z0, p.z);
                    st.z1 = std::max(st.z1, p.z);
                    st.y0 = std::min(st.y0, p.y);
                    st.y1 = std::max(st.y1, p.y);
                    st.x0 = std::min(st.x0, p.x);
                    st.x1 = std::max(st.x1, p.x);

                    for (const auto &d: nbr) {
                        int nx = p.x + d.x;
                        int ny = p.y + d.y;
                        int nz = p.z + d.z;
                        if (nz < z0 || nz > z1 || ny < y0 || ny > y1 || nx < x0 || nx > x1) continue;

                        int nid = lin(nz, ny, nx);
                        if (!mask[(size_t) nid] || vis[(size_t) nid]) continue;

                        vis[(size_t) nid] = 1;
                        stack.push_back(cv::Point3i(nx, ny, nz));
                    }
                }

                comps.push_back(st);
            }
        }
    }

    if (dbgVerbose && maskTot > 0) {
        double frac = (double)maskOn / (double)maskTot;
        // Print only for bbox-sized calls (heuristic): avoid flooding global calls.
        if (maskTot <= (size_t)120*120*20) {
            std::cout << "  [DBG][CCMask] threshLow=" << threshLow
                      << " maskOn=" << maskOn << "/" << maskTot
                      << " frac=" << frac
                      << " box(z,y,x)=(" << rz << "," << ry << "," << rx << ")"
                      << "\n";
        }
    }

    return comps;
}

EmbryoBrightTracker::CellState EmbryoBrightTracker::trackSingleCellByCCInBBox(
    int frameIdx,
    const std::vector<cv::Mat> &vol,
    const CellState &prev,
    float threshLow,
    bool &ok,
    std::vector<Comp3DStat> *outComps) const {
    ok = false;

    const BBox3D &b = prev.bbox;
    std::vector<Comp3DStat> comps = extractConnectedComponents3D(
        vol, threshLow,
        b.z0, b.z1, b.y0, b.y1, b.x0, b.x1,
        true /*use26*/);

    if (outComps) *outComps = comps;

    int maxV = 0;
    int sumV = 0;
    for (const auto& c : comps) { maxV = std::max(maxV, c.vox); sumV += c.vox; }

    if (dbgVerbose) {
        std::cout << "  [DBG][TrackCC] f=" << frameIdx << " id=" << prev.id
                  << " comps=" << comps.size()
                  << " maxVox=" << maxV
                  << " sumVox=" << sumV
                  << "\n";
    }

    if (comps.empty()) {

        CellState dead = prev;
        dead.alive = false;
        dead.lastSeenFrame = frameIdx;
        dead.voxelCount = 0;

        if (dbgVerbose) {
            std::cout << "  [DBG][TrackFail] f=" << frameIdx << " id=" << prev.id
              << " reason=comps_empty(threshLow_too_high_or_bbox_miss)"
              << "\n";
        }

        return dead;
    }

    // Filter tiny noise components aggressively inside bbox
    // Use matureDiamAvg as scale; if not ready, fallback.
    float d = (matureDiamAvg > 1e-3f) ? matureDiamAvg : 20.0f;
    float r = 0.5f * d;
    float matureV = (4.0f / 3.0f) * (float) M_PI * r * r * r;

    // Use prev.voxelCount as primary scale. Avoid hard floor that kills dim/fragmented daughters.
    int minVox = 30;  // was 120

    if (prev.voxelCount > 0) {
        // 2% of previous size is enough to keep dim daughters alive
        minVox = (int)std::lround((double)prev.voxelCount * 0.02);

        // soft bounds
        minVox = std::max(minVox, 30);
        minVox = std::min(minVox, std::max(80, (int)std::lround((double)prev.voxelCount * 0.25)));
    }

    std::vector<Comp3DStat> kept;
    kept.reserve(comps.size());

    for (const auto &c: comps) {
        if (c.vox >= minVox) kept.push_back(c);
    }

    if (dbgVerbose) {
        std::cout << "  [DBG][TrackFilter] f=" << frameIdx << " id=" << prev.id
              << " minVox=" << minVox
              << " kept=" << kept.size() << "/" << comps.size()
              << "\n";
    }

    if (kept.empty()) {
        // Fallback: do not kill track immediately; pick the largest component.
        int bestIdx = 0;
        for (int i = 1; i < (int)comps.size(); ++i) {
            if (comps[i].vox > comps[bestIdx].vox) bestIdx = i;
        }

        const auto &cst = comps[bestIdx];

        CellState cur = prev;
        cur.lastSeenFrame = frameIdx;
        cur.alive = true;
        cur.center = cst.center();
        cur.meanIntensity = cst.meanI();
        cur.voxelCount = cst.vox;
        cur.diameter = cst.diamXY();

        ok = true;

        if (dbgVerbose) {
            std::cout << "  [WARN][TrackFallback] f=" << frameIdx << " id=" << prev.id
                      << " kept_empty -> use_max_comp vox=" << cst.vox
                      << "\n";
        }

        return cur;
    }

    // Pick best component: nearest to prev.center, tie-break by larger vox
    int best = 0;
    float bestD2 = 1e30f;
    for (int i = 0; i < (int) kept.size(); ++i) {
        cv::Point3f ctr = kept[i].center();
        float d2 = dist2_3d(ctr, prev.center);
        if (d2 < bestD2 - 1e-6f) {
            bestD2 = d2;
            best = i;
        } else if (std::abs(d2 - bestD2) <= 1e-6f) {
            if (kept[i].vox > kept[best].vox) best = i;
        }
    }

    const auto &cst = kept[best];

    CellState cur = prev;
    cur.lastSeenFrame = frameIdx;
    cur.alive = true;
    cur.center = cst.center();
    cur.meanIntensity = cst.meanI();
    cur.voxelCount = cst.vox;

    cur.diameter = cst.diamXY();

    ok = true;
    return cur;
}

std::vector<EmbryoBrightTracker::CellState> EmbryoBrightTracker::detectInitialCellsGlobal(
    const std::vector<cv::Mat> &vol,
    float thresh) {
    int Z = (int) vol.size();
    int Y = vol[0].rows;
    int X = vol[0].cols;

    // Light denoise to reduce isolated sparkles
    std::vector<cv::Mat> work = vol;
    for (int z = 0; z < Z; ++z) {
        cv::GaussianBlur(work[z], work[z], cv::Size(0, 0), 1.0);
    }

    float threshLow = clampf(thresh * 0.40f, 0.03f, 0.95f);

    std::vector<Comp3DStat> comps = extractConnectedComponents3D(
        work, threshLow,
        0, Z - 1, 0, Y - 1, 0, X - 1,
        true /*use26*/);

    if (comps.empty()) return {};

    // Filter out tiny components (noise)
    // Use a conservative floor; later we will refine with bbox tracking.
    const int MIN_VOX_GLOBAL = 20000; // tune for your dataset scale; must kill debris
    std::vector<Comp3DStat> kept;
    kept.reserve(comps.size());
    for (const auto &c: comps) {
        if (c.vox >= MIN_VOX_GLOBAL) kept.push_back(c);
    }
    if (kept.empty()) return {};

    std::sort(kept.begin(), kept.end(), [](const Comp3DStat &a, const Comp3DStat &b) {
        return a.vox > b.vox; // volume priority; stable for embryo early stage
    });

    // Auto-select top K by volume gap rule (avoid hard-coding 4)
    // Keep components until volume drops below 0.35 of the largest.
    int K = 0;
    int v0 = kept[0].vox;
    for (int i = 0; i < (int) kept.size(); ++i) {
        if (kept[i].vox < (int) (0.35f * (float) v0)) break;
        K++;
        if (K >= 8) break; // safety cap
    }
    if (K <= 0) K = std::min(4, (int) kept.size()); // fallback

    std::vector<CellState> cells;
    cells.reserve(K);

    for (int i = 0; i < K; ++i) {
        const auto &st = kept[i];

        CellState c;
        c.id = std::to_string(i);
        c.alive = true;
        c.lastSeenFrame = 0;
        c.parentId = "";

        c.center = st.center();
        c.meanIntensity = st.meanI();
        c.voxelCount = st.vox;

        c.diameter = st.diamXY(); // stable diameter proxy (avoid voxelCount inflation)

        // temporary bbox; will be updated after matureDiamAvg is computed
        c.bbox = makeBBox(c.center, 20.0f, Z, Y, X);

        cells.push_back(c);
    }
    return cells;
}

EmbryoBrightTracker::CellState EmbryoBrightTracker::trackSingleCellInBBox(
    int frameIdx,
    const std::vector<cv::Mat> &vol,
    const CellState &prev,
    float thresh,
    bool &ok) const {
    ok = false;

    int Z = (int) vol.size();
    int Y = vol[0].rows;
    int X = vol[0].cols;

    const BBox3D &b = prev.bbox;

    double sumW = 0.0;
    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    double sumI = 0.0;
    int voxelCnt = 0;

    for (int z = b.z0; z <= b.z1; ++z) {
        const cv::Mat &m = vol[z];
        for (int y = b.y0; y <= b.y1; ++y) {
            const float *row = m.ptr<float>(y);
            for (int x = b.x0; x <= b.x1; ++x) {
                float v = row[x];
                if (v >= thresh) {
                    double w = (double) v;
                    sumW += w;
                    sumX += w * x;
                    sumY += w * y;
                    sumZ += w * z;
                    sumI += v;
                    voxelCnt++;
                }
            }
        }
    }

    CellState cur = prev;
    cur.lastSeenFrame = frameIdx;

    if (voxelCnt <= 0 || sumW <= 1e-9) {
        cur.alive = false;
        return cur;
    }

    cur.center.x = (float) (sumX / sumW);
    cur.center.y = (float) (sumY / sumW);
    cur.center.z = (float) (sumZ / sumW);
    cur.meanIntensity = (float) (sumI / (double) voxelCnt);
    cur.diameter = estimateDiameterFromVoxelCount(voxelCnt);
    cur.alive = true;

    ok = true;
    return cur;
}

// redudant
static void kmeans2_weighted(
    const std::vector<cv::Point3f> &pts,
    const std::vector<float> &w,
    cv::Point3f &c1,
    cv::Point3f &c2,
    int iters) {
    if (pts.empty()) return;

    c1 = pts.front();
    c2 = pts.back();

    for (int it = 0; it < iters; ++it) {
        double s1w = 0, s2w = 0;
        double s1x = 0, s1y = 0, s1z = 0;
        double s2x = 0, s2y = 0, s2z = 0;

        for (size_t i = 0; i < pts.size(); ++i) {
            const auto &p = pts[i];
            float ww = w[i];

            float d1 = (p.x - c1.x) * (p.x - c1.x) + (p.y - c1.y) * (p.y - c1.y) + (p.z - c1.z) * (p.z - c1.z);
            float d2 = (p.x - c2.x) * (p.x - c2.x) + (p.y - c2.y) * (p.y - c2.y) + (p.z - c2.z) * (p.z - c2.z);

            if (d1 <= d2) {
                s1w += ww;
                s1x += ww * p.x;
                s1y += ww * p.y;
                s1z += ww * p.z;
            } else {
                s2w += ww;
                s2x += ww * p.x;
                s2y += ww * p.y;
                s2z += ww * p.z;
            }
        }

        if (s1w > 1e-9) c1 = cv::Point3f((float) (s1x / s1w), (float) (s1y / s1w), (float) (s1z / s1w));
        if (s2w > 1e-9) c2 = cv::Point3f((float) (s2x / s2w), (float) (s2y / s2w), (float) (s2z / s2w));
    }
}

bool EmbryoBrightTracker::detectSplitInBBox(
    int frameIdx,
    const std::vector<cv::Mat> &vol,
    const CellState &parentPrev,
    float thresh,
    CellState &outC1,
    CellState &outC2) const {
    const BBox3D &b = parentPrev.bbox;

    std::vector<cv::Point3f> pts;
    std::vector<float> w;
    pts.reserve(200000);
    w.reserve(200000);

    int voxelCnt = 0;

    for (int z = b.z0; z <= b.z1; ++z) {
        const cv::Mat &m = vol[z];
        for (int y = b.y0; y <= b.y1; ++y) {
            const float *row = m.ptr<float>(y);
            for (int x = b.x0; x <= b.x1; ++x) {
                float v = row[x];
                if (v >= thresh) {
                    pts.push_back(cv::Point3f((float) x, (float) y, (float) z));
                    w.push_back(v);
                    voxelCnt++;
                }
            }
        }
    }

    if (voxelCnt < 200) return false;

    cv::Point3f c1, c2;
    kmeans2_weighted(pts, w, c1, c2, 10);

    float sep2 = (c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y) + (c1.z - c2.z) * (c1.z - c2.z);
    if (sep2 < 25.0f * 25.0f) return false;

    int cnt1 = 0, cnt2 = 0;
    double sum1I = 0, sum2I = 0;

    for (size_t i = 0; i < pts.size(); ++i) {
        const auto &p = pts[i];
        float d1 = (p.x - c1.x) * (p.x - c1.x) + (p.y - c1.y) * (p.y - c1.y) + (p.z - c1.z) * (p.z - c1.z);
        float d2 = (p.x - c2.x) * (p.x - c2.x) + (p.y - c2.y) * (p.y - c2.y) + (p.z - c2.z) * (p.z - c2.z);
        if (d1 <= d2) {
            cnt1++;
            sum1I += w[i];
        } else {
            cnt2++;
            sum2I += w[i];
        }
    }

    float expectedDaughterMinVox = 150.0f;

    if (cnt1 < expectedDaughterMinVox || cnt2 < expectedDaughterMinVox) return false;

    outC1 = parentPrev;
    outC2 = parentPrev;

    outC1.center = c1;
    outC2.center = c2;

    outC1.meanIntensity = (cnt1 > 0) ? (float) (sum1I / (double) cnt1) : 0.0f;
    outC2.meanIntensity = (cnt2 > 0) ? (float) (sum2I / (double) cnt2) : 0.0f;

    outC1.diameter = estimateDiameterFromVoxelCount(cnt1);
    outC2.diameter = estimateDiameterFromVoxelCount(cnt2);

    outC1.alive = true;
    outC2.alive = true;
    outC1.lastSeenFrame = frameIdx;
    outC2.lastSeenFrame = frameIdx;

    return true;
}

void EmbryoBrightTracker::printBBox(const CellState &c) const {
    const auto &b = c.bbox;
    std::cout << "  BBox(" << c.id << "): "
            << "z[" << b.z0 << "," << b.z1 << "] "
            << "y[" << b.y0 << "," << b.y1 << "] "
            << "x[" << b.x0 << "," << b.x1 << "]\n";
    std::cout << "    corners: "
            << "(" << b.x0 << "," << b.y0 << "," << b.z0 << ") "
            << "(" << b.x1 << "," << b.y0 << "," << b.z0 << ") "
            << "(" << b.x0 << "," << b.y1 << "," << b.z0 << ") "
            << "(" << b.x1 << "," << b.y1 << "," << b.z0 << ") "
            << "(" << b.x0 << "," << b.y0 << "," << b.z1 << ") "
            << "(" << b.x1 << "," << b.y0 << "," << b.z1 << ") "
            << "(" << b.x0 << "," << b.y1 << "," << b.z1 << ") "
            << "(" << b.x1 << "," << b.y1 << "," << b.z1 << ")\n";
}

void EmbryoBrightTracker::printFrameSummary(
    int frameIdx,
    const std::vector<CellState> &cells,
    float matureDiam) const {
    int aliveCnt = 0;
    for (const auto &c: cells) if (c.alive) aliveCnt++;

    std::cout << "==============================\n";
    std::cout << "[Frame " << frameIdx << "] total_alive=" << aliveCnt
            << " matureDiamAvg=" << matureDiam << "\n";

    if (frameIdx==0 || frameIdx==1) {
        std::cout << "  [GT] expected_alive≈4 (manual)\n";
    } else if (frameIdx==2) {
        std::cout << "  [GT] expected_alive≈4 (manual), 2 cells dimming pre-division\n";

    } else if (frameIdx==5) {
        std::cout << "  [GT] expected_alive≈6 (manual)\n";
    } else if (frameIdx==7) {
        std::cout << "  [GT] expected_alive≈7 (manual)\n";
    } else if (frameIdx==8 || frameIdx==9 || frameIdx==10) {
        std::cout << "  [GT] expected_alive≈7 (manual)\n";
    }

    if (frameIdx==5 && aliveCnt!=6) {
        std::cout << "  [ALERT] mismatch: expected 6 but got " << aliveCnt << "\n";
    }
    if (frameIdx==7 && aliveCnt!=7) {
        std::cout << "  [ALERT] mismatch: expected 7 but got " << aliveCnt << "\n";
    }
    if ((frameIdx==8||frameIdx==9||frameIdx==10) && aliveCnt!=7) {
        std::cout << "  [ALERT] mismatch: expected 7 but got " << aliveCnt << "\n";
    }

    std::cout << "  [DBG] Z_range_used=0.." << (int)34 << " (native), synth_out=255\n";

    int idx = 0;
    for (const auto &c: cells) {
        if (!c.alive) continue;
        std::cout << "  #" << idx << " id=" << c.id
                << " center(x,y,z)=(" << c.center.x << "," << c.center.y << "," << c.center.z << ")"
                << " diam=" << c.diameter
                << " meanI=" << c.meanIntensity
                << "\n";
        idx++;
    }

    for (const auto &c: cells) {
        if (!c.alive) continue;
        printBBox(c);
    }
}

void EmbryoBrightTracker::saveNewSynth(
    int frameIdx,
    const std::vector<cv::Mat> &realVol,
    const std::vector<CellState> &cells,
    float thresh) const {
    fs::path base(outDir);
    fs::path out = base / "new_synth" / std::to_string(frameIdx);
    fs::create_directories(out);

    int Z = (int) realVol.size();
    int Y = realVol[0].rows;
    int X = realVol[0].cols;

    std::vector<cv::Mat> synth;
    synth.reserve(Z);
    for (int z = 0; z < Z; ++z) {
        synth.push_back(cv::Mat::zeros(Y, X, CV_32F));
    }

    for (const auto &c: cells) {
        if (!c.alive) continue;
        const auto &b = c.bbox;
        for (int z = b.z0; z <= b.z1; ++z) {
            const cv::Mat &src = realVol[z];
            cv::Mat &dst = synth[z];
            for (int y = b.y0; y <= b.y1; ++y) {
                const float *srow = src.ptr<float>(y);
                float *drow = dst.ptr<float>(y);
                for (int x = b.x0; x <= b.x1; ++x) {
                    float v = srow[x];
                    if (v >= thresh) {
                        drow[x] = std::max(drow[x], v);
                    }
                }
            }
        }
    }

        // ---------- (1) Optional: keep native-Z synth for debugging ----------
    {
        fs::path outNative = base / "new_synth_native" / std::to_string(frameIdx);
        fs::create_directories(outNative);

        for (int z = 0; z < Z; ++z) {
            cv::Mat u8 = zSliceToU8(synth[z]);

            cv::Mat bgr;
            cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);

            for (const auto& c : cells) {
                if (!c.alive) continue;
                const auto& b = c.bbox;

                if (z >= b.z0 && z <= b.z1) {
                    drawDashedRect(
                        bgr,
                        cv::Rect(cv::Point(b.x0, b.y0), cv::Point(b.x1, b.y1)),
                        cv::Scalar(0, 255, 255),
                        2
                    );
                }

                int cz = (int)std::lround(c.center.z);
                if (z == cz) {
                    cv::circle(
                        bgr,
                        cv::Point((int)std::lround(c.center.x), (int)std::lround(c.center.y)),
                        4, cv::Scalar(255, 0, 255), -1
                    );

                    int half = (int)std::lround(0.5f * c.diameter);
                    int cx = (int)std::lround(c.center.x);
                    int cy = (int)std::lround(c.center.y);
                    cv::line(
                        bgr,
                        cv::Point(cx - half, cy),
                        cv::Point(cx + half, cy),
                        cv::Scalar(0, 255, 0), 2
                    );
                }
            }

            char buf[64];
            std::snprintf(buf, sizeof(buf), "z%03d.png", z);
            cv::imwrite((outNative / buf).string(), bgr);
        }
    }

    // ---------- (2) Required: stretched-Z synth to match real/ (ZT=255) ----------
    {
        // Keep your original path name: base/new_synth/<frame>/
        fs::path out255 = base / "new_synth" / std::to_string(frameIdx);
        fs::create_directories(out255);

        const int ZT = 255;
        for (int zt = 0; zt < ZT; ++zt) {
            float t = (ZT <= 1) ? 0.0f : (float)zt / (float)(ZT - 1);
            float zs = t * (float)(Z - 1);

            int z0 = (int)std::floor(zs);
            int z1 = std::min(Z - 1, z0 + 1);
            float a = zs - (float)z0;

            cv::Mat f0 = synth[z0];
            cv::Mat f1 = synth[z1];

            cv::Mat blend;
            if (z0 == z1) blend = f0;
            else cv::addWeighted(f0, 1.0 - a, f1, a, 0.0, blend);

            cv::Mat u8 = zSliceToU8(blend);

            cv::Mat bgr;
            cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);

            // Map bbox/center from native Z to ZT
            auto mapZ = [&](int zn) -> int {
                if (Z <= 1) return 0;
                float tt = (float)zn / (float)(Z - 1);
                return (int)std::lround(tt * (float)(ZT - 1));
            };
            auto mapZf = [&](float zn) -> int {
                if (Z <= 1) return 0;
                float tt = zn / (float)(Z - 1);
                return (int)std::lround(tt * (float)(ZT - 1));
            };

            for (const auto& c : cells) {
                if (!c.alive) continue;
                const auto& b = c.bbox;

                int bz0 = mapZ(b.z0);
                int bz1 = mapZ(b.z1);

                if (zt >= bz0 && zt <= bz1) {
                    drawDashedRect(
                        bgr,
                        cv::Rect(cv::Point(b.x0, b.y0), cv::Point(b.x1, b.y1)),
                        cv::Scalar(0, 255, 255),
                        2
                    );
                }

                int cz = mapZf(c.center.z);
                if (zt == cz) {
                    cv::circle(
                        bgr,
                        cv::Point((int)std::lround(c.center.x), (int)std::lround(c.center.y)),
                        4, cv::Scalar(255, 0, 255), -1
                    );

                    int half = (int)std::lround(0.5f * c.diameter);
                    int cx = (int)std::lround(c.center.x);
                    int cy = (int)std::lround(c.center.y);
                    cv::line(
                        bgr,
                        cv::Point(cx - half, cy),
                        cv::Point(cx + half, cy),
                        cv::Scalar(0, 255, 0), 2
                    );
                }
            }

            char buf[64];
            std::snprintf(buf, sizeof(buf), "z%03d.png", zt);
            cv::imwrite((out255 / buf).string(), bgr);
        }
    }
}

void EmbryoBrightTracker::saveRealStretchedZ255(
    int frameIdx,
    const std::vector<cv::Mat>& realVol) const
{
    fs::path base(outDir);
    fs::path out = base / "real" / std::to_string(frameIdx);
    fs::create_directories(out);

    int Z = (int)realVol.size();
    if (Z <= 0) return;

    int Y = realVol[0].rows;
    int X = realVol[0].cols;

    const int ZT = 255;
    for (int zt = 0; zt < ZT; ++zt) {
        float t = (ZT <= 1) ? 0.0f : (float)zt / (float)(ZT - 1);
        float zs = t * (float)(Z - 1);

        int z0 = (int)std::floor(zs);
        int z1 = std::min(Z - 1, z0 + 1);
        float a = zs - (float)z0;

        cv::Mat f0 = realVol[z0];
        cv::Mat f1 = realVol[z1];

        cv::Mat blend;
        if (z0 == z1) {
            blend = f0;
        } else {
            cv::addWeighted(f0, 1.0 - a, f1, a, 0.0, blend);
        }

        cv::Mat u8 = zSliceToU8(blend);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "z%03d.png", zt);
        cv::imwrite((out / buf).string(), u8);
    }
}

void EmbryoBrightTracker::updateViewer(
    int frameIdx,
    const std::vector<CellState> &cells) {
    std::vector<LineageTreeCreator::CellViz> viz;
    for (const auto &c: cells) {
        if (!c.alive) continue;
        LineageTreeCreator::CellViz v;
        v.rawName = c.id;
        v.x = c.center.x;
        v.y = c.center.y;
        viz.push_back(v);
    }
    viewer.update(frameIdx, viz);
}

void EmbryoBrightTracker::run(const std::vector<fs::path> &imagePaths) {
    if (imagePaths.empty()) {
        throw std::runtime_error("No frames provided.");
    }

    std::vector<cv::Mat> vol0 = loadVolume(imagePaths[0]);
    int Z = (int) vol0.size();
    int Y = vol0[0].rows;
    int X = vol0[0].cols;

    float globalThresh = percentileThreshold(vol0, 99.3f);
    globalThresh = clampf(globalThresh, 0.2f, 0.95f);

    std::vector<CellState> cells = detectInitialCellsGlobal(vol0, globalThresh);

    // Use stable diameter estimate from initial CC stats (diamXY) to avoid voxel-count inflation.
    float diamSum = 0.0f;
    int diamCnt = 0;
    for (auto &c: cells) {
        if (!c.alive) continue;
        if (c.diameter > 1e-3f) {
            diamSum += c.diameter;
            diamCnt++;
        }
    }

    matureDiamAvg = (diamCnt > 0) ? (diamSum / (float)diamCnt) : 40.0f;
    matureDiamAvg = clampf(matureDiamAvg, 60.0f, 95.0f);

    for (auto &c: cells) {
        c.bbox = makeBBox(c.center, matureDiamAvg, Z, Y, X);
    }

    printFrameSummary(0, cells, matureDiamAvg);

    saveRealStretchedZ255(0, vol0);
    saveNewSynth(0, vol0, cells, globalThresh);

    updateViewer(0, cells);

    for (int f = 1; f < (int) imagePaths.size(); ++f) {
        std::vector<cv::Mat> vol = loadVolume(imagePaths[f]);

        float pct = (f <= 7) ? 99.3f : 98.8f;
        float thresh = percentileThreshold(vol, pct);

        // Light denoise per-slice to reduce fragmentation inside bboxes
        std::vector<cv::Mat> volBlur = vol;
        for (auto &sl : volBlur) {
            cv::GaussianBlur(sl, sl, cv::Size(0, 0), 1.0);
        }

        thresh = clampf(thresh, 0.2f, 0.95f);
        float threshLowFrame      = clampf(thresh * 0.50f, 0.06f, 0.95f);

        float threshSplitLowFrame = clampf(thresh * 0.60f, 0.08f, 0.95f);

        std::cout << "[DBG][FrameThresh] f=" << f
                  << " thresh=" << thresh
                  << " threshLow=" << threshLowFrame
                  << " threshSplitLow=" << threshSplitLowFrame
                  << "\n";

        std::vector<CellState> nextCells;

        auto pushUnique = [&](const CellState& cand) {
            // if two tracks collapse to same blob, keep the one with larger voxelCount
            float boxDiam = clampf(cand.diameter, 30.0f, 70.0f);
            float mergeR2 = (0.25f * boxDiam) * (0.25f * boxDiam);

            for (auto& exist : nextCells) {
                if (!exist.alive) continue;
                if (dist2_3d(exist.center, cand.center) <= mergeR2) {
                    if (cand.voxelCount > exist.voxelCount) {
                        exist = cand;
                    }
                    return;
                }
            }
            nextCells.push_back(cand);
        };

        for (const auto &prev: cells) {
            if (!prev.alive) continue;

            float threshLow = threshLowFrame;

            bool ok = false;
            std::vector<Comp3DStat> compsInBox;

            CellState tracked = trackSingleCellByCCInBBox(f, vol, prev, threshLow, ok, &compsInBox);

            if (dbgVerbose) {
                std::cout << "  [DBG] frame=" << f << " id=" << prev.id
                          << " thresh=" << thresh
                          << " threshLow=" << threshLow
                          << " bboxZ=" << (prev.bbox.z1 - prev.bbox.z0 + 1)
                          << " bboxY=" << (prev.bbox.y1 - prev.bbox.y0 + 1)
                          << " bboxX=" << (prev.bbox.x1 - prev.bbox.x0 + 1)
                          << " compsInBox=" << compsInBox.size()
                          << " prevVox=" << prev.voxelCount
                          << "\n";
            }

            const auto& bb = prev.bbox;
            bool touch = (bb.z0==0 || bb.z1==Z-1 || bb.y0==0 || bb.y1==Y-1 || bb.x0==0 || bb.x1==X-1);
            std::cout << "  [DBG][BBox] f=" << f << " id=" << prev.id
                      << " touchEdge=" << (touch ? 1 : 0)
                      << " dzToEdge=(" << (prev.center.z - bb.z0) << "," << (bb.z1 - prev.center.z) << ")"
                      << " dyToEdge=(" << (prev.center.y - bb.y0) << "," << (bb.y1 - prev.center.y) << ")"
                      << " dxToEdge=(" << (prev.center.x - bb.x0) << "," << (bb.x1 - prev.center.x) << ")"
                      << "\n";

            // --- Split cooldown: avoid chain false splits ---
            const int SPLIT_COOLDOWN = 8;
            if (f - prev.lastSplitFrame < SPLIT_COOLDOWN) {
                float boxDiam = clampf(prev.diameter, 30.0f, 70.0f);
                tracked.bbox = makeBBox(tracked.center, boxDiam,
                                        (int)vol.size(), vol[0].rows, vol[0].cols);
                pushUnique(tracked);
                continue;
            }

            float threshSplitLow = clampf(thresh * 0.60f, 0.08f, 0.95f);

            std::vector<Comp3DStat> compsForSplit = extractConnectedComponents3D(
                vol, threshSplitLow,
                prev.bbox.z0, prev.bbox.z1, prev.bbox.y0, prev.bbox.y1, prev.bbox.x0, prev.bbox.x1,
                true /*use26*/);

            if (!ok) {

                pendingSplits.erase(prev.id);

                LostEvent le;
                le.frame = f;
                le.cellId = prev.id;
                le.lastCenter = prev.center;
                lostEvents.push_back(le);

                std::cout << ">>> Lost Cell frame=" << f
                        << " id=" << prev.id
                        << " lastCenter=(" << prev.center.x << "," << prev.center.y << "," << prev.center.z << ")"
                        << "\n";
                continue;
            }

            // *******************************************************************************************
            // Decide split by CC count inside bbox (structure change 1 to 2)
            std::vector<Comp3DStat> bigComps;
            bigComps.reserve(compsInBox.size());

            // Volume-based noise filter for split decision
            float d = (matureDiamAvg > 1e-3f) ? matureDiamAvg : 20.0f;
            float r = 0.5f * d;
            float matureV = (4.0f / 3.0f) * (float) M_PI * r * r * r;

            int maxVSplit = 0;
            int sumVSplit = 0;
            for (const auto& c : compsForSplit) {
                maxVSplit = std::max(maxVSplit, c.vox);
                sumVSplit += c.vox;
            }

            int minDaughterVox = 200;
            if (maxVSplit > 0) {
                // each daughter should be at least a fraction of the biggest blob
                minDaughterVox = std::max(minDaughterVox, (int)std::lround(maxVSplit * 0.22));
            }
            if (sumVSplit > 0) {
                // avoid sum being inflated by fragments
                minDaughterVox = std::min(minDaughterVox, (int)std::lround(sumVSplit * 0.12));
            }
            // hard safety bounds
            minDaughterVox = std::max(minDaughterVox, 200);

            if (dbgVerbose) {
                std::cout << "  [DBG][SplitCC] f=" << f << " id=" << prev.id
                          << " compsForSplit=" << compsForSplit.size()
                          << " maxVox=" << maxVSplit
                          << " sumVox=" << sumVSplit
                          << " minDaughterVox=" << minDaughterVox
                          << "\n";
            }

            for (const auto &cst: compsForSplit) {
                if (cst.vox >= minDaughterVox) bigComps.push_back(cst);
            }

            if ((int)bigComps.size() < 2) {
                std::cout << "  [DBG][SplitFail] f=" << f << " id=" << prev.id
                          << " reason=bigComps_lt_2"
                          << " bigComps=" << bigComps.size()
                          << " (likely minDaughterVox_too_high_or_fragmentation)"
                          << "\n";
            }

            std::sort(bigComps.begin(), bigComps.end(), [](const Comp3DStat &a, const Comp3DStat &b) {
                return a.vox > b.vox;
            });

            bool split = false;
            CellState c1, c2;

            // ************************** split event check  **************************
            if ((int) bigComps.size() >= 2) {
                cv::Point3f p1 = bigComps[0].center();
                cv::Point3f p2 = bigComps[1].center();

                // distance between two daughter centers
                float sep2 = dist2_3d(p1, p2);

                // near parent center test
                float nearR2 = d * d; // within mature diameter
                bool near1 = dist2_3d(p1, prev.center) <= nearR2;
                bool near2 = dist2_3d(p2, prev.center) <= nearR2;

                // --- Neighbor exclusion: daughters should not sit near other cells' centers ---
                auto minDist2ToOther = [&](const cv::Point3f& q) -> float {
                    float best = 1e30f;
                    for (const auto& other : cells) {
                        if (!other.alive) continue;
                        if (other.id == prev.id) continue;
                        best = std::min(best, dist2_3d(q, other.center));
                    }
                    return best;
                };

                // use ONE boxDiam for all scale rules in this block
                float boxDiam = clampf(prev.diameter, 30.0f, 90.0f);

                float avoidR2 = (0.55f * boxDiam) * (0.55f * boxDiam);
                bool farFromOthers1 = (minDist2ToOther(p1) >= avoidR2);
                bool farFromOthers2 = (minDist2ToOther(p2) >= avoidR2);

                // separation test using boxDiam scale
                float minSep = 0.30f * boxDiam;
                bool separated = (sep2 >= (minSep * minSep));

                std::cout << "  [DBG][SplitCond] f=" << f << " id=" << prev.id
                          << " nearR=" << d
                          << " near1=" << (near1?1:0) << " d1=" << std::sqrt(dist2_3d(p1, prev.center))
                          << " near2=" << (near2?1:0) << " d2=" << std::sqrt(dist2_3d(p2, prev.center))
                          << " separated=" << (separated?1:0) << " sep=" << std::sqrt(sep2)
                          << " minSep=" << minSep
                          << " farOther1=" << (farFromOthers1?1:0)
                          << " farOther2=" << (farFromOthers2?1:0)
                          << "\n";

                // ******************************* make the split decision **********************
                // 【SPLIT DECISION RULE】:
                // We declare a real cell division only when ALL of following geometric conditions are satisfied:
                //
                // 1。near1 && near2
                //    Both candidate daughter centers (p1 and p2) must lie close to the previous parent center.
                //    The distance must be within one mature cell diameter. This ensures the split happens locally
                //    around the parent and not somewhere else inside the bounding box.
                //
                // 2.  separated
                //    The 2 candidate daughter centers must be sufficiently separated from each other.
                //    If the distance between them is too small, the structure is likely still one single cell
                //    (for example a dim or elongated "pancake" shape before full division).
                //
                // 3. farFromOthers1 && farFromOthers2
                //    Each candidate daughter must also be sufficiently far from every OTHER cell center
                //    in the current frame. This prevents false splits caused by neighboring cells whose
                //    bright regions accidentally fall inside the parent's bounding box.
                //
                // Only when ALL these constraints hold simultaneously do we consider the structure to be
                // a true biological cell division and create two new child cells from the parent.
                if (near1 && near2 && separated && farFromOthers1 && farFromOthers2) {
                    split = true;

                    c1 = prev;
                    c2 = prev;
                    c1.center = p1;
                    c2.center = p2;

                    c1.voxelCount = bigComps[0].vox;
                    c2.voxelCount = bigComps[1].vox;

                    c1.meanIntensity = bigComps[0].meanI();
                    c2.meanIntensity = bigComps[1].meanI();

                    c1.diameter = estimateDiameterFromVoxelCount(c1.voxelCount);
                    c2.diameter = estimateDiameterFromVoxelCount(c2.voxelCount);

                    c1.alive = true;
                    c2.alive = true;
                    c1.lastSeenFrame = f;
                    c2.lastSeenFrame = f;

                    c1.lastSplitFrame = f;
                    c2.lastSplitFrame = f;
                }
            }

            // ---------- Pending split confirm (2-frame rule) ----------
            // If we see a split once, do NOT commit immediately.
            // Commit only if the split condition is stable in the next frame
            // and the two centers are consistent (allow swap).
            auto centersConsistent = [&](const cv::Point3f &a1, const cv::Point3f &a2,
                                         const cv::Point3f &b1, const cv::Point3f &b2,
                                         float tol) -> bool {
                float tol2 = tol * tol;
                float d11 = dist2_3d(a1, b1) + dist2_3d(a2, b2);
                float d12 = dist2_3d(a1, b2) + dist2_3d(a2, b1);
                return (std::min(d11, d12) <= 2.0f * tol2);
            };

            auto &ps = pendingSplits[prev.id];

            if (split) {
                float boxDiam = clampf(prev.diameter, 30.0f, 90.0f);
                float tol = 0.35f * boxDiam; // center stability tolerance

                if (!ps.active) {
                    // first time seeing a split -> mark pending, do NOT commit
                    ps.active = true;
                    ps.firstFrame = f;
                    ps.c1 = c1.center; // current candidate
                    ps.c2 = c2.center;

                    std::cout << "  [DBG][SplitPending] f=" << f << " id=" << prev.id
                              << " store_centers=(" << ps.c1.x << "," << ps.c1.y << "," << ps.c1.z << ")-("
                              << ps.c2.x << "," << ps.c2.y << "," << ps.c2.z << ")\n";

                    split = false; // force fallthrough to normal tracking this frame
                } else {
                    // already pending, only confirm on next frame
                    if (f == ps.firstFrame + 1 &&
                        centersConsistent(ps.c1, ps.c2, c1.center, c2.center, tol)) {
                        std::cout << "  [DBG][SplitConfirm] f=" << f << " id=" << prev.id
                                  << " confirmed_from_f=" << ps.firstFrame << "\n";

                        ps.active = false; // commit now (keep split=true)
                        ps.firstFrame = -1;
                    } else {
                        // not confirmed: refresh pending to current frame, do NOT commit
                        std::cout << "  [DBG][SplitPendingRefresh] f=" << f << " id=" << prev.id
                                  << " prev_pending_f=" << ps.firstFrame << "\n";

                        ps.active = true;
                        ps.firstFrame = f;
                        ps.c1 = c1.center;
                        ps.c2 = c2.center;
                        split = false;
                    }
                }
            } else if (ps.active) {
                // If no split is seen this frame, clear stale pending state.
                ps.active = false;
                ps.firstFrame = -1;
            }
            // ---------- end pending split confirm ----------

            if (!split) {
                float boxDiam = clampf(prev.diameter, 30.0f, 70.0f);
                tracked.bbox = makeBBox(tracked.center, boxDiam, (int)vol.size(), vol[0].rows, vol[0].cols);

                pushUnique(tracked);
                continue;
            }

            // Split event
            splitCounter++;
            c1.id = prev.id + ".1";
            c2.id = prev.id + ".2";
            c1.parentId = prev.id;
            c2.parentId = prev.id;

            float boxDiam = clampf(prev.diameter, 30.0f, 70.0f);
            c1.bbox = makeBBox(c1.center, boxDiam, (int)vol.size(), vol[0].rows, vol[0].cols);
            c2.bbox = makeBBox(c2.center, boxDiam, (int)vol.size(), vol[0].rows, vol[0].cols);

            pushUnique(c1);
            pushUnique(c2);

            SplitEvent ev;
            ev.frame = f;
            ev.parent = prev.id;
            ev.child1 = c1.id;
            ev.child2 = c2.id;
            ev.child1Diameter = c1.diameter;
            ev.child2Diameter = c2.diameter;
            splitEvents.push_back(ev);

            std::cout << ">>> Split Event #" << splitCounter
                    << " frame=" << f
                    << " parent=" << ev.parent
                    << " children=(" << ev.child1 << "," << ev.child2 << ")"
                    << " childDiam=(" << ev.child1Diameter << "," << ev.child2Diameter << ")"
                    << " childVox=(" << c1.voxelCount << "," << c2.voxelCount << ")"
                    << "\n";
        }

        printFrameSummary(f, nextCells, matureDiamAvg);

        saveRealStretchedZ255(f, vol);
        saveNewSynth(f, vol, nextCells, thresh);

        updateViewer(f, nextCells);

        cells = nextCells;
    }
}
