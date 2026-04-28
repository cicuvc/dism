function get_cugencodes(target)
    local cugencodes = {}
    for i, v in pairs(target:get("cugencodes")) do
        if string.startswith(v, "sm_") then
            table.insert(cugencodes, v)
        end
        if v == "native" then
            for j, vf in pairs(target:get("cu_native_archs")) do
                table.insert(cugencodes, vf)
            end
        end
    end
    cugencodes = table.unique(cugencodes)
    return cugencodes
end
function make_dev_compile_args(v, objfile, sourcefile, target)
    local device_compile_args = {"--cuda-device-only", "-c", "-o", objfile, sourcefile, string.format("--cuda-gpu-arch=%s", v), "-v"}
    
    for j, vf in pairs(target:get("cuflags")) do
        if not (vf == "-G") then
            table.insert(device_compile_args, vf)
        end
    end
    for j, vf in pairs(target:get("includedirs")) do
        table.insert(device_compile_args, "-I"..vf)
    end
    return device_compile_args
end
function build_batchcmds(target, batchcmds, sourcefile, opt, skip_dep)
    import("core.project.depend")
    import("core.base.option")
    
    local cc = target:tool("cc")
    -- print("Use cc " .. cc)
    -- print("make batchcmds for " .. sourcefile)
    
    local objectfile = target:objectfile(sourcefile)
    local depend_file = target:dependfile(objectfile)
    local cxx_depend_file = depend_file .. ".dep"
    -- local dependinfo = option.get("rebuild") and {} or (depend.load(depend_file) or {})
    -- print(dependinfo)
    local ref_files = {}
    if not skip_dep then
        os.mkdir(path.directory(cxx_depend_file))
        
        for i, v in pairs(get_cugencodes(target)) do
            if string.startswith(v, "sm_") then
                
                local device_compile_args = make_dev_compile_args(v, "tmp.o", sourcefile, target)
                table.join2(device_compile_args, {"-MM", "-MF", cxx_depend_file})
                if os.isfile(cxx_depend_file) then
                    depend.on_changed(function() os.runv(cc, device_compile_args) end, {files = {sourcefile}})
                else
                    os.runv(cc, device_compile_args)
                end
                
                local depinfo = {}
                depinfo.depfiles_format = "gcc"
                depinfo.depfiles = io.readfile(cxx_depend_file)
                depend.save(depinfo, cxx_depend_file .. ".alt")
                local dep_files = depend.load(cxx_depend_file .. ".alt").files
                table.join2(ref_files, dep_files)
                break
            end
        end
    end
    
    -- invoke compiler
    local cc = target:tool("cc")
    local fatbin = target:tool("fatbin")
    table.insert(target:objectfiles(), objectfile)
    local device_objdir = path.directory(objectfile)
    local device_fatbin = path.join(device_objdir, path.basename(objectfile) .. ".gpucode.fatbin")
    batchcmds:mkdir(device_objdir)

    -- Detect CUDA version from toolchain
    local cuda_major = 0
    for _, toolchain_inst in ipairs(target:toolchains()) do
        local version = toolchain_inst:get("cuda_major")
        if version then
            cuda_major = version
            break
        end
    end

    local fatbin_args = {"-64", "--create", device_fatbin}
    
    for i, v in pairs(get_cugencodes(target)) do
        if string.startswith(v, "sm_") then
            local objfile = path.join(device_objdir, string.format("%s.gpucode.%s.o", path.basename(objectfile), v))
            local device_compile_args = make_dev_compile_args(v, objfile, sourcefile, target)
            --batchcmds:show("[ %d%%] compiling CUDA file %s - %s", opt.progress, sourcefile, v)
            batchcmds:show_progress(opt.progress, "${color.build.object}compiling CUDA file %s - %s", sourcefile, v)
            batchcmds:mkdir(path.directory(objfile))
            batchcmds:vrunv(cc, device_compile_args, {stderr = io.stderr, stdout = io.stdout})

            -- Use new format for CUDA 13+
            if cuda_major >= 13 then
                table.insert(fatbin_args, string.format("--image3=kind=elf,sm=%s,file=%s", v:sub(4), objfile))
            else
                table.insert(fatbin_args, string.format("--image=profile=%s,file=%s", v, objfile))
            end
        end
    end
    
    batchcmds:vrunv(fatbin, fatbin_args,  {stderr = io.stderr, stdout = io.stdout})
    local host_compile_args = {"--cuda-host-only", "-c", "-o", objectfile, "-Xclang", "-fcuda-include-gpubinary", "-Xclang", device_fatbin, sourcefile, "-fPIC"}
    for j, vf in pairs(target:get("cuflags")) do
        if not (vf == "-G") then
            table.insert(host_compile_args, vf)
        end
    end
    for j, vf in pairs(target:get("includedirs")) do
        table.insert(host_compile_args, "-I"..vf)
    end
    for j, vf in pairs(target:get("cxxflags")) do
        table.insert(host_compile_args, vf)
    end
    batchcmds:vrunv(cc, host_compile_args,  {stderr = io.stderr, stdout = io.stdout})
    
                
    -- batchcmds:vrunv("gcc", {"-o", objectfile, "-c", sourcefile})
    table.sort(ref_files)
    while table.find(ref_files,"\\") do
        table.remove(ref_files,table.find(ref_files,"\\")[1])
    end
    batchcmds:add_depfiles(ref_files)
    batchcmds:set_depmtime(os.mtime(objectfile))
    batchcmds:set_depcache(depend_file)
end