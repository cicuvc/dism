
toolchain("clang-nvptxex")
    set_kind("standalone")
    on_load(function (toolchain)
        import("lib.detect.find_program")
        import("core.base.option")
        import("core.project.config")
        import("detect.sdks.find_cuda")

        local cuda = find_cuda(nil, {verbose = true})

        local testfile = path.join(os.tmpdir(), "empty.cpp")
        os.runv("touch", {testfile})
        local program = find_program("clang", {paths = {"/usr/bin", "/usr/local/bin"}, check = function(program)  os.runv(program, {testfile, "-E", "-std=c++20"}) end })

        -- print("Use cxc " .. program)
        -- set toolset
        toolchain:set("toolset", "cc", program)
        toolchain:set("toolset", "cxx", program)
        toolchain:set("toolset", "ld", program)

        local fatbin = find_program("fatbinary", { paths = {"/usr/bin", "/usr/local/bin", cuda.bindir} })
        toolchain:set("toolset", "fatbin", fatbin)

        -- Detect CUDA version from fatbinary
        local outdata, errdata = os.iorunv(fatbin, {"--version"})
        local cuda_version = 0
        if outdata then
            -- Parse version string like "fatbinary 12.8" or "fatbinary 13.1"
            local major, minor = outdata:match("release%s+(%d+)%.(%d+)")
            if major then
                cuda_version = tonumber(major) * 10 + tonumber(minor)
                toolchain:set("cuda_version", cuda_version)
                toolchain:set("cuda_major", tonumber(major))
            end
        end
    end)



rule("cuda.clang.gencodes")
    set_extensions(".cu")
    add_deps("mode.debug", "mode.release")

    on_config(function (target)

        import("core.platform.platform")
        import("lib.detect.find_cudadevices")
        import("core.base.hashset")
        import("core.tool.compiler")

        local cuda_envs
        for _, toolchain_inst in ipairs(target:toolchains()) do
            if toolchain_inst:name() == "cuda" then
                cuda_envs = toolchain_inst:runenvs()
                break
            end
        end
        local devices = find_cudadevices({skip_compute_mode_prohibited = true, order_by_flops = true, envs = cuda_envs, plat = target:plat(), arch = target:arch()})
        local archs = {}
        for i, v in ipairs(devices) do
            table.insert(archs, string.format("sm_%d%d", v.major, v.minor))
        end
        
        archs = table.unique(archs)
        target:set("cu_native_archs", archs)


        local optimize = target:get("optimize")
        if optimize then
            local optimize_flags = compiler.map_flags("cxx", "optimize", optimize)
            
            target:add("cuflags", optimize_flags)
        end
       
    end)

    on_clean(function (target)
        for i, sourcefile in ipairs(target:sourcefiles()) do
            local depend_file = path.join(target:dependir(), sourcefile .. ".dep")
            if os.isfile(depend_file) then
                os.rm(depend_file)
            end

            local objectfile = target:objectfile(sourcefile)
            local device_objdir = path.directory(objectfile)
            local objfile = path.join(device_objdir, string.format("%s.gpucode.*.o", path.basename(objectfile)))

            for _, filepath in ipairs(os.files(objfile)) do
                os.rm(filepath)
            end
        end
    end)

    on_build_files(function (target, jobgraph, sourcebatch, opt)
        import("private.utils.batchcmds")
        import("clangcuda")


        for i, sourcefile in ipairs(sourcebatch.sourcefiles) do
            jobgraph:add("job/" .. sourcefile, function(index, total, opt) 
                local cmds = batchcmds.new({target = target})
                clangcuda.build_batchcmds(target, cmds, sourcefile, opt)
                cmds:runcmds(opt) 
            end)
        end
    end, {jobgraph = true, batch = true})

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        import("clangcuda").build_batchcmds(target, batchcmds, sourcefile, opt, true)
    end)

rule_end()


rule("cuda")
    add_deps("cuda.clang.gencodes")

    on_config(function (target)
        import("detect.sdks.find_cuda")

        local cuda = find_cuda(nil, {verbose = true})

        target:add("linkdirs", cuda.linkdirs)
        target:add("links", "cuda", "cudart")
    end)
    
    on_link(function (target)
        import("core.project.depend")
        import("utils.progress")

        local objectfiles = target:objectfiles()
        local targetfile = target:targetfile()

        local invoke_link = function ()
            local link = target:tool("ld")
            

            if not os.isdir(path.directory(targetfile)) then
                os.mkdir(path.directory(targetfile))
            end

            -- progress.show(opt.progress, "${color.build.object}linking target file %s", path.basename(targetfile))
            
            local linkargs = {"-o", targetfile }
            linkargs = table.move(objectfiles, 1, #objectfiles, #linkargs + 1, linkargs)


            for i,v in pairs(target:get("linkdirs")) do
                table.insert(linkargs, string.format("-L%s", v))
            end

            for i,v in pairs(target:get("links")) do
                table.insert(linkargs, string.format("-l%s", v))
            end

            for i,v in pairs(target:get("culdflags")) do
                table.insert(linkargs, v)
            end

            os.vrunv(link, linkargs,  {stderr = io.stderr, stdout = io.stdout})

        end

        if os.isfile(targetfile) then
            depend.on_changed(invoke_link, {files = objectfiles})
        else
            invoke_link()
        end
    end)
