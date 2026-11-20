# Διεργασία εξαγωγής βίντεο: τρέχει από τον Live Service σε ξεχωριστή διεργασία Julia (όπως ο sim_worker).
# Διαβάζει meta.json / heightmap.png / cm_k.png / frames.bin ενός ολοκληρωμένου project και γράφει animation.mp4 μέσω ffmpeg.
# Χρήση:  julia --project=webapp webapp/video_worker.jl <φάκελος project> [fps]
using JSON3, Images, FileIO

const TARGET_SIDE = 900                                    # επιθυμητή μεγαλύτερη διάσταση του βίντεο σε pixels (το grid μεγεθύνεται με ακέραιο παράγοντα)
const CM_ALPHA = (0.25, 0.55)                              # όπως στο frontend: διαφάνεια = 0.25 + 0.55*συγκέντρωση

# ffmpeg: το πακέτο FFMPEG_jll αν υπάρχει στο περιβάλλον, αλλιώς το ffmpeg του συστήματος
function ffmpeg_cmd()
    try
        @eval using FFMPEG_jll
        return Base.invokelatest(() -> FFMPEG_jll.ffmpeg())
    catch
        exe = Sys.which("ffmpeg")
        exe === nothing && error("δεν βρέθηκε ffmpeg (ούτε FFMPEG_jll ούτε στο PATH)")
        return `$exe`
    end
end

write_progress(dir, status, frame, T; extra...) = open(joinpath(dir, "video_progress.json"), "w") do io
    JSON3.write(io, (; status, frame, T, extra...))
end

# 5x3 γραμματοσειρά ψηφίων για το χρονόμετρο και τον μετρητή διαφυγόντων
const DIGITS = Dict(
    '0' => ["111", "101", "101", "101", "111"], '1' => ["010", "110", "010", "010", "111"], '2' => ["111", "001", "111", "100", "111"],
    '3' => ["111", "001", "111", "001", "111"], '4' => ["101", "101", "111", "001", "001"], '5' => ["111", "100", "111", "001", "111"],
    '6' => ["111", "100", "111", "101", "111"], '7' => ["111", "001", "010", "010", "010"], '8' => ["111", "101", "111", "101", "111"],
    '9' => ["111", "101", "111", "001", "111"], ':' => ["000", "010", "000", "010", "000"], '/' => ["001", "001", "010", "100", "100"])

@inline function setpx!(buf, W, H, x, y, c)
    (1 <= x <= W && 1 <= y <= H) || return
    o = ((y - 1) * W + (x - 1)) * 3
    @inbounds begin buf[o+1] = c[1]; buf[o+2] = c[2]; buf[o+3] = c[3] end
end

function draw_text!(buf, W, H, x0, y0, str, k, color)      # k = μεγέθυνση κάθε "pixel" της γραμματοσειράς
    x = x0
    for ch in str
        glyph = get(DIGITS, ch, nothing)
        if glyph !== nothing
            for (gy, row) in enumerate(glyph), (gx, bit) in enumerate(row)
                bit == '1' || continue
                for dy in 1:k, dx in 1:k
                    setpx!(buf, W, H, x + (gx - 1) * k + dx, y0 + (gy - 1) * k + dy, color)
                end
            end
        end
        x += 4k
    end
end

function fill_disc!(buf, W, H, cx, cy, r, color, edge)
    for y in floor(Int, cy - r):ceil(Int, cy + r), x in floor(Int, cx - r):ceil(Int, cx + r)
        d = hypot(x - 0.5 - cx, y - 0.5 - cy)
        d <= r && setpx!(buf, W, H, x, y, d > r - 0.8 && r > 2 ? edge : color)
    end
end

function ring!(buf, W, H, cx, cy, r, color)
    for a in range(0, 2π; length = max(24, ceil(Int, 6r)))
        setpx!(buf, W, H, round(Int, cx + r * cos(a)) + 1, round(Int, cy + r * sin(a)) + 1, color)
    end
end

person_color(tl, escaped) = escaped ? (0x7f, 0x8c, 0x9a) : tl >= 3 ? (0xff, 0x3b, 0x30) : tl <= 1 ? (0x34, 0xc7, 0x59) : (0xff, 0x9f, 0x0a)

function render_video(dir, fps)
    meta = JSON3.read(read(joinpath(dir, "meta.json"), String))
    rows, cols, n, stride = meta.rows, meta.cols, meta.n_agents, meta.stride
    hm = load(joinpath(dir, "heightmap.png"))
    gray = UInt8.(round.(clamp.(Float64.(Gray.(hm)), 0, 1) .* 255))          # [γραμμή, στήλη]
    s = max(1, round(Int, TARGET_SIDE / max(rows, cols)))
    W, H = cols * s, rows * s
    W2, H2 = iseven(W) ? W : W + 1, iseven(H) ? H : H + 1                    # το yuv420p θέλει ζυγές διαστάσεις
    up(v, y, x) = v[cld(y, s), cld(x, s)]

    # φόντο ανά CM: heightmap σε γκρι + το CM ως κίτρινο→κόκκινο overlay (όπως στον browser)
    bgs = map(1:meta.num_cms) do k
        cm = Float64.(Gray.(load(joinpath(dir, "cm_$k.png"))))
        bg = zeros(UInt8, W2 * H2 * 3)
        for y in 1:H, x in 1:W
            g = Float64(up(gray, y, x)); v = up(cm, y, x)
            r, gg, b = g, g, g
            if v >= 0.02
                a = CM_ALPHA[1] + CM_ALPHA[2] * v
                r = (1 - a) * g + a * 255; gg = (1 - a) * g + a * 235 * (1 - v); b = (1 - a) * g
            end
            o = ((y - 1) * W2 + (x - 1)) * 3
            bg[o+1] = round(UInt8, clamp(r, 0, 255)); bg[o+2] = round(UInt8, clamp(gg, 0, 255)); bg[o+3] = round(UInt8, clamp(b, 0, 255))
        end
        bg
    end

    per = n * stride
    nframes = min(meta.T, filesize(joinpath(dir, "frames.bin")) ÷ (4 * per))
    nframes >= 1 || error("δεν υπάρχουν frames")
    mpp = meta.meters_per_pixel
    r_agent = max(0.5 / mpp * s, 1.6)                                          # δίσκος 1 m, με ελάχιστη ακτίνα ώστε να φαίνεται
    out = joinpath(dir, "animation.mp4")
    tmp = joinpath(dir, "animation_tmp.mp4")
    cmd = `$(ffmpeg_cmd()) -y -loglevel error -f rawvideo -pix_fmt rgb24 -s $(W2)x$(H2) -r $fps -i - -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart $tmp`
    write_progress(dir, "running", 0, nframes)
    data = Mmap_read(joinpath(dir, "frames.bin"), nframes * per)
    buf = similar(bgs[1])
    k_scale = max(1, round(Int, H2 / 150))                                    # μέγεθος κειμένου ~ ανάλογο της ανάλυσης
    process = open(cmd, "w")
    try
        for f in 1:nframes
            t = f - 1
            idx = max(1, searchsortedlast(Vector{Float64}(meta.cm_switch_times), t))
            copyto!(buf, bgs[idx])
            for e in meta.goals
                ring!(buf, W2, H2, e.col * s, e.row * s, meta.goal_radius * s, (0xff, 0x3b, 0x30))
            end
            esc = 0
            base = (f - 1) * per
            for k in 0:n-1
                o = base + k * stride
                col, row, tl, state = data[o+1], data[o+2], data[o+3], data[o+4]
                state == 1 && (esc += 1)
                fill_disc!(buf, W2, H2, col * s, row * s, r_agent, person_color(tl, state == 1), (0x00, 0x00, 0x00))
                state == 2 && ring!(buf, W2, H2, col * s, row * s, max(2.6r_agent, 5), (0x22, 0xd3, 0xee))
            end
            m, sec = divrem(f, 60)
            draw_text!(buf, W2, H2, 8, 8, string(lpad(m, 2, '0'), ':', lpad(sec, 2, '0')), k_scale, (0xff, 0xff, 0xff))
            draw_text!(buf, W2, H2, 8, 8 + 7k_scale, string(esc, '/', n), k_scale, (0x7f, 0xd8, 0x9a))
            write(process, buf)
            f % 50 == 0 && write_progress(dir, "running", f, nframes)
        end
        close(process)
        wait(process)
        success(process) || error("το ffmpeg απέτυχε")
    catch
        try kill(process) catch end
        rm(tmp; force = true)
        rethrow()
    end
    mv(tmp, out; force = true)
    write_progress(dir, "done", nframes, nframes; bytes = filesize(out))
end

# frames.bin ως Float32 (ένα read· 2500 frames × 100 agents × 16 B ≈ 4 MB)
function Mmap_read(file, count)
    v = Vector{Float32}(undef, count)
    open(file) do io
        read!(io, v)
    end
    v
end

dir = ARGS[1]
fps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
try
    render_video(dir, fps)
catch e
    write_progress(dir, "failed", 0, 0; error = sprint(showerror, e))
    showerror(stderr, e, catch_backtrace())
    exit(1)
end
