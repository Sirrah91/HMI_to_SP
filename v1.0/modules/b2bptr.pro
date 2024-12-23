; use .r and copy the code to the SSW IDL terminal
; .r
; otherwise type .r b2bprt.pro (or .run)

; Define the parent directory path
parent_directory = '/nfsscratch/david/NN/data/SDO_HMI_dconS/'

; Find all subfolders
subfolders = FILE_SEARCH(parent_directory + '*')

; Iterate over each subfolder file
for isub = 0, N_ELEMENTS(subfolders) - 1 do begin

    ; Define the folder path
    folder_path = subfolders[isub] + '/'

    ; Find all files with the "*.field.fits" ending
    field_files = FILE_SEARCH(folder_path + '*[.]field[.]fits')

    ; Iterate over each field file
    for i = 0, N_ELEMENTS(field_files) - 1 do begin 
        ; Extract the base name without the extension
        base_name = FILE_BASENAME(field_files[i])

        ; Split the base name by dots to extract the prefix
        prefix = STRSPLIT(base_name, '.', /EXTRACT)

        ; Construct the corresponding file array
        prefix[3] = 'inclination'
        incl_file = folder_path + STRJOIN(prefix, '.')
        prefix[3] = 'azimuth'
        azimuth_file = folder_path + STRJOIN(prefix, '.')
        prefix[3] = 'disambig'
        disambig_file = folder_path + STRJOIN(prefix, '.')
        files = [field_files[i], incl_file, azimuth_file, disambig_file]

        read_sdo, files[2], index, azi, /uncomp_delete
        read_sdo, files[3], index, ambig, /uncomp_delete
        hmi_disambig, azi, ambig, 1
        read_sdo, files[0:2], index, data, /uncomp_delete

        if (N_ELEMENTS(index) GT 0) AND (N_ELEMENTS(index.history) GT 0) THEN BEGIN
            ; Find occurrences of "rotated" in index.history
            rotated_indices = WHERE(STRPOS(index.history, "rotated") NE -1, count)

            ; If "rotated" is found in any element, add 180 to azi
            if (count GT 0) THEN BEGIN
                data[*,*,2] = azi + 180  ; to compensate the frame rotation done by im_patch
            ENDIF
        ENDIF

        hmi_b2ptr, index[0], data, bptr, lonlat=lonlat

        prefix[3] = 'ptr'
        SAVE, bptr, lonlat, filename=folder_path + STRJOIN(prefix[0:3], '.') + '.sav'
    endfor
endfor
end
